"""Cross backbone for the DiCM experiments."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

from dicm.utils.artifacts import verify_base

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dicm.module.backbone import load_backbone  # noqa: E402
from dicm.module.backbone_partition import lock_from_scores  # noqa: E402
from dicm.module.concept_bench import (  # noqa: E402
    Bench, neutral_prompts, prompts_for, retain_captions, save_json, sha,
)
from dicm.module.conflict_bench import weight_override  # noqa: E402
from dicm.module.sparse_delta import compose, dense_weights, disabled, export, project, set_mode, sparse_delta_supports  # noqa: E402

CHAIN = ["cat", "horse", "dog", "elephant"]
CFG = dict(lr=5e-4, eta=3.0, budget_frac=0.054, retain_weight=8.0, retain_batch=8, cohesion_weight=4.0,
           cohesion_batch=2, interaction_weight=4.0, accept_round=100, accept_max=600, accept_rate=0.83,
           subset_prob=0.5, t_min=20, t_max=980, scale=4096.0, lock_fraction=0.5, lock_draws=16,
           train_seed=17, cache_steps=10)
ESD = dict(steps=200, lr={"sd15": 1e-5, "sdxl": 1e-6}, eta=3.0, ddim_steps=20)
UCE = dict(lam=1.0, anchor="a photograph", prefixes=["", "image of ", "photo of "],
           preserve=["a person", "a car", "a tree", "a flower", "a bicycle", "a boat", "a bird", "a sofa",
                     "a bed", "an apple", "a bottle", "a banana", "a building", "a table"])
UCE_ERASE = {c: [c, f"a photo of a {c}"] for c in CHAIN}


def mse(a, b):
    return F.mse_loss(a.float(), b.float())


def noisy(bb, x0, t, gen):
    return bb.add_noise(x0, torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=gen), t)


def retain_lock(bb, layers, latents, conds, retain_train, cfg, log):
    net = bb.denoiser
    params = {n: p for n, p in net.named_parameters() if n in layers}
    for p in params.values():
        p.requires_grad_(True)
    acc = {n: torch.zeros(p.shape, device=p.device, dtype=torch.float32) for n, p in params.items()}
    gen = torch.Generator(device=bb.device).manual_seed(cfg["train_seed"] + 99)
    for i in range(cfg["lock_draws"]):
        prompt = retain_train[i % len(retain_train)]
        t = torch.randint(cfg["t_min"], cfg["t_max"] + 1, (1,), device=bb.device, generator=gen)
        pred = bb.predict(noisy(bb, latents[prompt], t, gen), t, conds[prompt])
        v = torch.randn(pred.shape, device=pred.device, dtype=torch.float32, generator=gen)
        (cfg["scale"] * (pred.float() * (v / v.norm())).sum()).backward()
        for n, p in params.items():
            acc[n] += p.grad.abs().float() / cfg["scale"]
            p.grad = None
    for p in params.values():
        p.requires_grad_(False)
    locks = lock_from_scores(acc, cfg["lock_fraction"])
    info = dict(locked=sum(int(l.sum()) for l in locks.values()), total=sum(l.numel() for l in locks.values()))
    log(f"lock {info}")
    return locks, info


def library_tensors(mods, payloads):
    out = []
    for p in payloads:
        d = {}
        for name, m in mods.items():
            t = torch.zeros(m.delta.shape, device=m.delta.device, dtype=torch.float32)
            t.reshape(-1)[p[name]["indices"].to(t.device)] = p[name]["values"].to(t.device, torch.float32)
            d[name] = t
        out.append(d)
    return out


def set_library(mods, tensors, picks):
    with torch.no_grad():
        for name, m in mods.items():
            m.library.zero_()
            for i in picks:
                m.library.add_(tensors[i][name])


def train_module(bb, layers, concept, latents, conds, retain_train, library, locks, cfg, log, bench, budget):
    net = bb.denoiser
    idx = CHAIN.index(concept)
    gen = torch.Generator(device=bb.device).manual_seed(cfg["train_seed"] + idx)
    rng = random.Random(cfg["train_seed"] + 7 * idx)
    own = prompts_for(concept, "train")
    neutral = neutral_prompts()
    lib_concepts = list(library["concepts"]) if library else []
    use_inter = bool(lib_concepts)
    null = conds[""]
    history, accepted = [], None
    best = dict(rate=-1.0, step=0, payload=None)
    with sparse_delta_supports(net, layers) as mods:
        tensors = library_tensors(mods, library["payloads"]) if use_inter else []
        set_library(mods, tensors, range(len(tensors)) if use_inter else [])
        params = [m.delta for m in mods.values()]
        opt = torch.optim.Adam(params, lr=cfg["lr"])

        def backward(loss):
            (cfg["scale"] * loss).backward()
            return float(loss.detach())

        step = 0
        while step < cfg["accept_max"]:
            t = torch.randint(cfg["t_min"], cfg["t_max"] + 1, (1,), device=bb.device, generator=gen)
            p_c = rng.choice(own)
            x_c = noisy(bb, latents[p_c], t, gen)
            r_prompts = rng.sample(retain_train, cfg["retain_batch"])
            x_r = [noisy(bb, latents[p], t, gen) for p in r_prompts]
            x_n = [(p, noisy(bb, latents[p], t, gen)) for p in rng.sample(neutral, cfg["cohesion_batch"])]
            picks, x_i = [], []
            if use_inter:
                n = len(tensors)
                picks = rng.sample(range(n), rng.randint(1, n - 1)) if (n > 1 and rng.random() < cfg["subset_prob"]) else list(range(n))
                set_library(mods, tensors, picks)
                lc = lib_concepts[rng.choice(picks)]
                p_l = rng.choice(prompts_for(lc, "train"))
                x_i = [(p_c, x_c), (r_prompts[0], x_r[0]), (p_l, noisy(bb, latents[p_l], t, gen))]
            with torch.no_grad():
                with disabled(mods):
                    e0 = bb.predict(x_c, t, null if isinstance(null, dict) else null).float()
                    ec = bb.predict(x_c, t, conds[p_c]).float()
                    T = dict(effect=e0 - cfg["eta"] * (ec - e0),
                             retain=[bb.predict(x, t, conds[p]).float() for x, p in zip(x_r, r_prompts)],
                             neutral=[bb.predict(x, t, conds[p]).float() for p, x in x_n],
                             f0=[bb.predict(x, t, conds[p]).float() for p, x in x_i])
                if use_inter:
                    set_mode(mods, "b")
                    T["fL"] = [bb.predict(x, t, conds[p]).float() for p, x in x_i]
            opt.zero_grad(set_to_none=True)
            rec = dict(step=step, t=int(t), lib=len(picks))
            set_mode(mods, "a")
            rec["effect"] = backward(mse(bb.predict(x_c, t, conds[p_c]).float(), T["effect"]))
            rec["retain"] = 0.0
            for x, p, tr in zip(x_r, r_prompts, T["retain"]):
                rec["retain"] += backward(cfg["retain_weight"] * mse(bb.predict(x, t, conds[p]).float(), tr) / cfg["retain_batch"])
            rec["cohesion"] = 0.0
            for (p, x), tn in zip(x_n, T["neutral"]):
                rec["cohesion"] += backward(cfg["cohesion_weight"] * mse(bb.predict(x, t, conds[p]).float(), tn) / cfg["cohesion_batch"])
            if use_inter:
                rec["interaction"] = 0.0
                for (p, x), f0, fL in zip(x_i, T["f0"], T["fL"]):
                    set_mode(mods, "a")
                    fM = bb.predict(x, t, conds[p]).float()
                    set_mode(mods, "union")
                    fLM = bb.predict(x, t, conds[p]).float()
                    rec["interaction"] += backward(cfg["interaction_weight"] * (((fLM - fL) - (fM - f0)) ** 2).mean() / len(x_i))
                    del fM, fLM
                set_mode(mods, "a")
            for p in params:
                if p.grad is not None:
                    p.grad.div_(cfg["scale"])
                    assert torch.isfinite(p.grad).all(), (concept, step)
            rec["gnorm"] = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            rec.update(project(mods, budget, locks))
            history.append(rec)
            step += 1
            if step % 50 == 0 or step == 1:
                log(f"  {concept} step {step} " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}))
            if step % cfg["accept_round"] == 0:
                set_mode(mods, "a")
                rate = bench.validation_rate(concept)
                log(f"  {concept} acceptance after {step}: {rate:.2f}")
                history.append(dict(step=step, acceptance=rate))
                if rate > best["rate"]:
                    best = dict(rate=rate, step=step, payload=export(mods))
                if rate >= cfg["accept_rate"]:
                    accepted = dict(steps=step, rate=rate)
                    break
        if accepted is None:
            set_mode(mods, "a")
            final = bench.validation_rate(concept)
            log(f"  {concept} acceptance at limit {step}: {final:.2f}")
            if final > best["rate"]:
                best = dict(rate=final, step=step, payload=export(mods))
            # acceptance is not monotone in training steps on every backbone, so ship the best
            # state the compiler observed rather than the last one
            accepted = dict(steps=best["step"], rate=best["rate"], final_rate=final,
                            final_steps=step, exhausted=True, used_best=True)
        payload = best["payload"] if accepted.get("used_best") else export(mods)
    return payload, history, accepted


# ---------------------------------------------------------------- baselines
def train_esdx(bb, concept, log):
    """ESD-x: fine-tune the cross-attention weights toward a negative-guidance target.

    x_t is produced by rolling the current model out along the DDIM trajectory for a random
    number of steps, so the (x_t, t) pair the loss sees is on-distribution. Sampling pure noise
    and pairing it with a late timestep diverges on the larger backbones.
    """
    net = bb.denoiser
    frozen = copy.deepcopy(net).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    for p in net.parameters():
        p.requires_grad_(False)
    params = []
    for n, p in net.named_parameters():
        if "attn2" in n:
            p.requires_grad_(True)
            params.append(p)
    lr = ESD["lr"][bb.key] if isinstance(ESD["lr"], dict) else ESD["lr"]
    opt = torch.optim.Adam(params, lr=lr)
    with torch.no_grad():
        cond_c = bb.encode(prompts_for(concept, "train")[0])
        cond_0 = bb.encode("")
    g = torch.Generator(device=bb.device).manual_seed(CHAIN.index(concept))
    sched = bb.pipe.scheduler
    from diffusers import DDIMScheduler

    ddim = DDIMScheduler.from_config(sched.config)
    ddim.set_timesteps(ESD["ddim_steps"], device=bb.device)
    t0 = time.time()
    for step in range(1, ESD["steps"] + 1):
        n_steps = int(torch.randint(1, ESD["ddim_steps"], (1,), generator=g, device=bb.device))
        z = torch.randn(1, bb.latent_channels, bb.latent_size, bb.latent_size, generator=g,
                        device=bb.device, dtype=bb.dtype)
        with torch.no_grad():
            for i in range(n_steps):
                t = ddim.timesteps[i].reshape(1)
                z = ddim.step(bb.predict(z, t, cond_c), t.item(), z).prev_sample
            t_cur = ddim.timesteps[min(n_steps, ESD["ddim_steps"] - 1)].reshape(1)
            e0 = frozen_predict(bb, frozen, z, t_cur, cond_0)
            ec = frozen_predict(bb, frozen, z, t_cur, cond_c)
            target = e0 - ESD["eta"] * (ec - e0)
        loss = F.mse_loss(bb.predict(z, t_cur, cond_c).float(), target.float())
        assert torch.isfinite(loss), f"esd {concept} diverged at step {step}"
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            log(f"  esd {concept} step {step} loss={loss.item():.5f}")
    for p in params:
        p.requires_grad_(False)
    del frozen, opt
    torch.cuda.empty_cache()
    return time.time() - t0


def frozen_predict(bb, frozen, x, t, cond):
    """bb.predict against a detached copy of the denoiser."""
    real = bb.denoiser
    holder = bb.pipe
    name = "unet" if hasattr(holder, "unet") else "transformer"
    setattr(holder, name, frozen)
    try:
        return bb.predict(x, t, cond)
    finally:
        setattr(holder, name, real)


@torch.no_grad()
def run_uce(bb, concepts, layers, log):
    """Closed-form UCE on the backbone's cross-attention K/V projections."""
    t0 = time.time()
    erase = [pre + e for c in concepts for e in UCE_ERASE[c] for pre in UCE["prefixes"]]

    E = torch.stack([bb.kv_context(p) for p in erase], 1)
    A = torch.stack([bb.kv_context(UCE["anchor"]) for _ in erase], 1)
    P = torch.stack([bb.kv_context(p) for p in UCE["preserve"]], 1)
    params = dict(bb.denoiser.named_parameters())
    out = {}
    for name in layers:
        W = params[name].data.float()
        d = E.shape[0]
        G = E @ E.T + P @ P.T + UCE["lam"] * torch.eye(d, device=E.device)
        out[name] = ((W @ A) @ E.T + (W @ P) @ P.T + UCE["lam"] * W) @ torch.linalg.inv(G)
    log(f"  uce {concepts} {time.time() - t0:.1f}s over {len(out)} tensors")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["sd15", "sdxl"])
    ap.add_argument("--arms", nargs="+", default=["ours", "esd_seq", "esd_merge", "uce_seq", "uce_joint"])
    ap.add_argument("--concepts", nargs="+", default=CHAIN)
    ap.add_argument("--budget-frac", type=float, default=CFG["budget_frac"])
    ap.add_argument("--lr", type=float, default=CFG["lr"])
    ap.add_argument("--accept-max", type=int, default=CFG["accept_max"])
    ap.add_argument("--accept-round", type=int, default=CFG["accept_round"])
    ap.add_argument("--retain-weight", type=float, default=CFG["retain_weight"])
    ap.add_argument("--cohesion-weight", type=float, default=CFG["cohesion_weight"])
    ap.add_argument("--eta", type=float, default=CFG["eta"], help="negative-guidance strength of the effect term")
    ap.add_argument("--esd-lr", type=float, default=None, help="override the per-backbone ESD baseline learning rate")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    chain = args.concepts
    logf = open(out / "run.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    torch.set_num_threads(4)
    cfg = dict(CFG, budget_frac=args.budget_frac, lr=args.lr, accept_max=args.accept_max,
               accept_round=args.accept_round, retain_weight=args.retain_weight,
               cohesion_weight=args.cohesion_weight, eta=args.eta)
    if args.esd_lr is not None:
        ESD["lr"] = args.esd_lr
    caps = retain_captions()
    bb = load_backbone(args.backbone)
    layers = bb.kv_layers()
    stats = bb.kv_stats()
    budget = int(args.budget_frac * stats["coordinates"])
    log(f"backbone {args.backbone} {stats} budget {budget}")
    before = bb.fingerprint()
    verify_base(out, before, bb.denoiser.dtype)
    save_json(out / "protocol.json", dict(experiment="cross_backbone", backbone=args.backbone, cfg=cfg, esd=ESD, uce=UCE,
                                          chain=chain, kv=stats, budget=budget, retain=caps,
                                          hashes={p: sha(ROOT / p) for p in ["dicm/experiments/cross_backbone.py",
                                                                             "dicm/module/backbone.py"]}))
    bench = Bench(bb.pipe, out, caps["test"], chain, backbone=bb)
    log("preparing base images")
    save_json(out / "base_summary.json", bench.prepare_base())
    assert bb.fingerprint() == before

    cache = out / "training_cache.pt"
    prompts = sorted({p for c in chain for w in ("train",) for p in prompts_for(c, w)} | set(caps["train"]) | set(neutral_prompts()))
    if cache.exists():
        latents = torch.load(cache, weights_only=True, map_location=bb.device)["latents"]
    else:
        log(f"caching {len(prompts)} latents")
        latents = bb.cache_latents(prompts, cfg["cache_steps"], cfg["train_seed"])
        torch.save(dict(latents=latents), cache)
    with torch.no_grad():
        conds = {p: bb.encode(p) for p in prompts + [""]}
    assert bb.fingerprint() == before

    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    acceptance = {}

    def evaluate(tag, weights, concs):
        if tag in results:
            log(f"skip {tag}")
            return
        log(f"EVAL {tag}")
        t0 = time.time()
        with weight_override(bb.denoiser, weights):
            res = bench.evaluate(tag, concs)
        assert bb.fingerprint() == before
        res["seconds"] = time.time() - t0
        results[tag] = res
        save_json(results_path, results)
        log("  " + json.dumps({c: (None if v["success"] is None else round(v["success"], 2)) for c, v in res["erasure"].items()})
            + f" retain {res['retain']['dino']:.3f}/{res['retain']['clip']:.3f}")

    if "ours" in args.arms:
        lock_path = out / "retain_lock.pt"
        if lock_path.exists():
            locks = {n: t.to(bb.device) for n, t in torch.load(lock_path, weights_only=True, map_location="cpu").items()}
        else:
            locks, info = retain_lock(bb, layers, latents, conds, caps["train"], cfg, log)
            torch.save({n: t.cpu() for n, t in locks.items()}, lock_path)
            save_json(out / "retain_lock.json", info)
        assert bb.fingerprint() == before
        compiled = []
        for c in chain:
            path = out / "modules" / f"{c}.pt"
            if path.exists():
                bank = torch.load(path, weights_only=True, map_location="cpu")
                log(f"reuse ours/{c}")
            else:
                library = None if not compiled else dict(concepts=[x for x, _ in compiled], payloads=[b for _, b in compiled])
                log(f"TRAIN ours/{c} library={[x for x, _ in compiled]}")
                t0 = time.time()
                bank, hist, acc = train_module(bb, layers, c, latents, conds, caps["train"], library, locks, cfg, log, bench, budget)
                assert bb.fingerprint() == before
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(bank, path)
                save_json(out / "history" / f"{c}.json", dict(seconds=time.time() - t0, history=hist, acceptance=acc))
                acceptance[c] = acc
                save_json(out / "acceptance.json", acceptance)
            compiled.append((c, bank))
            evaluate(f"ours/{c}", dense_weights(bb.denoiser, bank), [c])
            if len(compiled) >= 2:
                members = [x for x, _ in compiled]
                evaluate("ours/" + "+".join(members), dense_weights(bb.denoiser, compose([b for _, b in compiled])), members)

    if set(args.arms) & {"esd_seq", "esd_merge", "uce_seq", "uce_joint"}:
        # baseline fine-tuning needs fp32 master weights; fp16 Adam on the raw weights diverges
        log("loading fp32 backbone for the baselines")
        bb32 = load_backbone(args.backbone, dtype=torch.float32)
        net = bb32.denoiser
        base_state = {n: p.detach().clone().cpu() for n, p in net.named_parameters()}
        names = [n for n, _ in net.named_parameters() if "attn2" in n]
        store_path = out / "baseline_deltas.pt"
        store = torch.load(store_path, weights_only=True, map_location="cpu") if store_path.exists() else {}

        def reset(delta=None):
            with torch.no_grad():
                params = dict(net.named_parameters())
                for n, p in params.items():
                    p.data.copy_(base_state[n].to(p.device))
                for n, d in (delta or {}).items():
                    params[n].data.add_(d.to(params[n].device, params[n].dtype))

        def capture():
            cur = dict(net.named_parameters())
            o = {}
            for n in names:
                d = cur[n].detach().float().cpu() - base_state[n].float()
                if bool((d != 0).any()):
                    o[n] = d.to(torch.float16)
            return o

        for i, c in enumerate(chain):
            if f"esd/{c}" not in store:
                log(f"TRAIN esd single {c}")
                reset()
                train_esdx(bb32, c, log)
                store[f"esd/{c}"] = capture()
                torch.save(store, store_path)
            tag = "esd/seq_" + "_".join(chain[:i + 1])
            if tag not in store:
                log(f"TRAIN {tag}")
                reset(store["esd/seq_" + "_".join(chain[:i])] if i else None)
                train_esdx(bb32, c, log)
                store[tag] = capture()
                torch.save(store, store_path)
            tag = "uce/seq_" + "_".join(chain[:i + 1])
            if tag not in store:
                reset(store["uce/seq_" + "_".join(chain[:i])] if i else None)
                w = run_uce(bb32, [c], layers, log)
                store[tag] = {n: (v.float().cpu() - base_state[n].float()).to(torch.float16) for n, v in w.items()}
                torch.save(store, store_path)
            tag = "uce/joint_" + "_".join(chain[:i + 1])
            if tag not in store:
                reset()
                w = run_uce(bb32, chain[:i + 1], layers, log)
                store[tag] = {n: (v.float().cpu() - base_state[n].float()).to(torch.float16) for n, v in w.items()}
                torch.save(store, store_path)
        reset()
        del bb32, net
        torch.cuda.empty_cache()
        assert bb.fingerprint() == before
        params = dict(bb.denoiser.named_parameters())

        def apply(delta):
            return {n: (params[n].detach().float() + v.to(bb.device).float()).to(params[n].dtype) for n, v in delta.items()}

        def merged(ds):
            o = {}
            for d in ds:
                for n, v in d.items():
                    o[n] = o.get(n, 0) + v.float()
            return o

        for i, c in enumerate(chain):
            members = chain[:i + 1]
            key = "_".join(members)
            if "esd_seq" in args.arms:
                evaluate(f"esd_seq/{key}", apply(store["esd/seq_" + key]), members)
            if "esd_merge" in args.arms:
                evaluate(f"esd_merge/{key}", apply(merged([store[f"esd/{m}"] for m in members])), members)
            if "uce_seq" in args.arms:
                evaluate(f"uce_seq/{key}", apply(store["uce/seq_" + key]), members)
            if "uce_joint" in args.arms:
                evaluate(f"uce_joint/{key}", apply(store["uce/joint_" + key]), members)
    save_json(out / "final_status.json", dict(experiment="cross_backbone", backbone=args.backbone, complete=True, chain=chain,
                                              budget=budget, kv=stats, fingerprint_before=before,
                                              fingerprint_after=bb.fingerprint()))
    log("COMPLETE")


if __name__ == "__main__":
    main()
