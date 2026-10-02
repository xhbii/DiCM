"""Cross domain for the DiCM experiments."""
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
from dicm.models.sd15_wrapper import load_sd15_pipeline  # noqa: E402
from dicm.module.backbone_partition import lock_from_scores  # noqa: E402
from dicm.module.concept_bench import (  # noqa: E402
    Bench, fill, kv_layers, neutral_prompts, prompts_for, retain_captions, save_json, sha,
)
from dicm.module.conflict_bench import fingerprint, weight_override  # noqa: E402
from dicm.module.latents import _add_noise, _embed, cache_prompt_latents  # noqa: E402
from dicm.module.sparse_delta import compose, dense_weights, disabled, export, project, set_mode, sparse_delta_supports  # noqa: E402

OUT = ROOT / "outputs/cross_domain"
CHAIN = ["van_gogh", "monet", "picasso", "hokusai", "nudity", "cat"]
CFG = dict(lr=5e-4, eta=3.0, budget=800_000, retain_weight=8.0, retain_batch=8, cohesion_weight=4.0, cohesion_batch=2,
           interaction_weight=4.0, accept_round=100, accept_max=600, accept_rate=0.83, subset_prob=0.5,
           t_min=20, t_max=980, scale=4096.0, lock_fraction=0.5, lock_draws=16, train_seed=17, cache_steps=10)
ESD = dict(steps=200, lr=1e-5, eta=3.0, ddim_steps=20)
UCE = dict(lam=1.0, anchor="a photograph", prefixes=["", "image of ", "photo of ", "painting of "],
           preserve=["a person", "a car", "a tree", "a flower", "a bicycle", "a boat", "a bird", "a sofa", "a bed",
                     "an apple", "a bottle", "a banana", "a building", "a table"])
UCE_ERASE = {"van_gogh": ["Van Gogh style", "a painting by Van Gogh"], "monet": ["Monet style", "a painting by Monet"],
             "picasso": ["Picasso style", "a painting by Picasso"], "hokusai": ["Hokusai style", "a painting by Hokusai"],
             "nudity": ["nudity", "a nude person", "naked"], "cat": ["cat", "a photo of a cat"]}


def noisy(pipe, x0, t, gen):
    return _add_noise(pipe.scheduler, x0, torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=gen), t)


def mse(a, b):
    return F.mse_loss(a.float(), b.float())


def retain_lock(pipe, layers, latents, embeddings, retain_train, cfg):
    unet = pipe.unet
    params = {n: p for n, p in unet.named_parameters() if n in layers}
    for p in params.values():
        p.requires_grad_(True)
    acc = {n: torch.zeros(p.shape, device=p.device, dtype=torch.float32) for n, p in params.items()}
    gen = torch.Generator(device="cuda").manual_seed(cfg["train_seed"] + 99)
    for i in range(cfg["lock_draws"]):
        prompt = retain_train[i % len(retain_train)]
        t = torch.randint(cfg["t_min"], cfg["t_max"] + 1, (1,), device="cuda", generator=gen)
        pred = unet(noisy(pipe, latents[prompt], t, gen), t, encoder_hidden_states=embeddings[prompt]).sample
        v = torch.randn(pred.shape, device=pred.device, dtype=torch.float32, generator=gen)
        (cfg["scale"] * (pred.float() * (v / v.norm())).sum()).backward()
        for n, p in params.items():
            acc[n] += p.grad.abs().float() / cfg["scale"]
            p.grad = None
    for p in params.values():
        p.requires_grad_(False)
    locks = lock_from_scores(acc, cfg["lock_fraction"])
    return locks, dict(locked=sum(int(l.sum()) for l in locks.values()), total=sum(l.numel() for l in locks.values()))


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


def train_module(pipe, layers, concept, latents, embeddings, retain_train, library, locks, cfg, log, bench):
    unet = pipe.unet
    idx = CHAIN.index(concept)
    gen = torch.Generator(device="cuda").manual_seed(cfg["train_seed"] + idx)
    rng = random.Random(cfg["train_seed"] + 7 * idx)
    own = prompts_for(concept, "train")
    neutral = neutral_prompts()
    lib_concepts = list(library["concepts"]) if library else []
    use_inter = bool(lib_concepts)
    null = embeddings[""]
    history, accepted = [], None
    with sparse_delta_supports(unet, layers) as mods:
        tensors = library_tensors(mods, library["payloads"]) if use_inter else []
        set_library(mods, tensors, range(len(tensors)) if use_inter else [])
        params = [m.delta for m in mods.values()]
        opt = torch.optim.Adam(params, lr=cfg["lr"])

        def backward(loss):
            (cfg["scale"] * loss).backward()
            return float(loss.detach())

        step = 0
        while step < cfg["accept_max"]:
            t = torch.randint(cfg["t_min"], cfg["t_max"] + 1, (1,), device="cuda", generator=gen)
            p_c = rng.choice(own)
            x_c = noisy(pipe, latents[p_c], t, gen)
            r_prompts = rng.sample(retain_train, cfg["retain_batch"])
            x_r = [noisy(pipe, latents[p], t, gen) for p in r_prompts]
            x_n = [(p, noisy(pipe, latents[p], t, gen)) for p in rng.sample(neutral, cfg["cohesion_batch"])]
            picks, x_i = [], []
            if use_inter:
                n = len(tensors)
                picks = rng.sample(range(n), rng.randint(1, n - 1)) if (n > 1 and rng.random() < cfg["subset_prob"]) else list(range(n))
                set_library(mods, tensors, picks)
                lc = lib_concepts[rng.choice(picks)]
                p_l = rng.choice(prompts_for(lc, "train"))
                x_i = [(p_c, x_c), (r_prompts[0], x_r[0]), (p_l, noisy(pipe, latents[p_l], t, gen))]
            with torch.no_grad():
                with disabled(mods):
                    e0 = unet(x_c, t, encoder_hidden_states=null).sample.float()
                    ec = unet(x_c, t, encoder_hidden_states=embeddings[p_c]).sample.float()
                    T = dict(effect=e0 - cfg["eta"] * (ec - e0),
                             retain=[unet(x, t, encoder_hidden_states=embeddings[p]).sample.float() for x, p in zip(x_r, r_prompts)],
                             neutral=[unet(x, t, encoder_hidden_states=embeddings[p]).sample.float() for p, x in x_n],
                             f0=[unet(x, t, encoder_hidden_states=embeddings[p]).sample.float() for p, x in x_i])
                if use_inter:
                    set_mode(mods, "b")
                    T["fL"] = [unet(x, t, encoder_hidden_states=embeddings[p]).sample.float() for p, x in x_i]
            opt.zero_grad(set_to_none=True)
            rec = dict(step=step, t=int(t), lib=len(picks))
            set_mode(mods, "a")
            pred = unet(x_c, t, encoder_hidden_states=embeddings[p_c]).sample.float()
            rec["effect"] = backward(mse(pred, T["effect"]))
            del pred
            rec["retain"] = 0.0
            for x, p, tr in zip(x_r, r_prompts, T["retain"]):
                pred = unet(x, t, encoder_hidden_states=embeddings[p]).sample.float()
                rec["retain"] += backward(cfg["retain_weight"] * mse(pred, tr) / cfg["retain_batch"])
                del pred
            rec["cohesion"] = 0.0
            for (p, x), tn in zip(x_n, T["neutral"]):
                pred = unet(x, t, encoder_hidden_states=embeddings[p]).sample.float()
                rec["cohesion"] += backward(cfg["cohesion_weight"] * mse(pred, tn) / cfg["cohesion_batch"])
                del pred
            if use_inter:
                rec["interaction"] = 0.0
                for (p, x), f0, fL in zip(x_i, T["f0"], T["fL"]):
                    set_mode(mods, "a")
                    fM = unet(x, t, encoder_hidden_states=embeddings[p]).sample.float()
                    set_mode(mods, "union")
                    fLM = unet(x, t, encoder_hidden_states=embeddings[p]).sample.float()
                    rec["interaction"] += backward(cfg["interaction_weight"] * (((fLM - fL) - (fM - f0)) ** 2).mean() / len(x_i))
                    del fM, fLM
                set_mode(mods, "a")
            for p in params:
                if p.grad is not None:
                    p.grad.div_(cfg["scale"])
                    assert torch.isfinite(p.grad).all(), (concept, step)
            rec["gnorm"] = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            rec.update(project(mods, cfg["budget"], locks))
            history.append(rec)
            step += 1
            if step % 50 == 0 or step == 1:
                log(f"  {concept} step {step} " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}))
            if step % cfg["accept_round"] == 0:
                set_mode(mods, "a")
                rate = bench.validation_rate(concept)
                log(f"  {concept} acceptance after {step}: {rate:.2f}")
                history.append(dict(step=step, acceptance=rate))
                if rate >= cfg["accept_rate"]:
                    accepted = dict(steps=step, rate=rate)
                    break
        if accepted is None:
            set_mode(mods, "a")
            accepted = dict(steps=step, rate=bench.validation_rate(concept), exhausted=True)
            log(f"  {concept} acceptance at limit {step}: {accepted['rate']:.2f}")
        payload = export(mods)
    return payload, history, accepted


# ---------------------------------------------------------------- baselines
def train_esdx(pipe, concept, log):
    unet = pipe.unet
    frozen = copy.deepcopy(unet).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    for p in unet.parameters():
        p.requires_grad_(False)
    params = []
    for n, p in unet.named_parameters():
        if "attn2" in n:
            p.requires_grad_(True)
            params.append(p)
    opt = torch.optim.Adam(params, lr=ESD["lr"])
    from diffusers import DDIMScheduler

    ddim = DDIMScheduler.from_config(pipe.scheduler.config)
    ddim.set_timesteps(ESD["ddim_steps"], device="cuda")
    with torch.no_grad():
        emb_c = _embed(pipe, prompts_for(concept, "train")[0]).detach()
        emb_null = _embed(pipe, "").detach()
    g = torch.Generator(device="cuda").manual_seed(CHAIN.index(concept))
    unet.train()
    t0 = time.time()
    for step in range(1, ESD["steps"] + 1):
        n_steps = int(torch.randint(1, ESD["ddim_steps"], (1,), generator=g, device="cuda"))
        z = torch.randn(1, 4, 64, 64, generator=g, device="cuda")
        with torch.no_grad():
            for i in range(n_steps):
                t = ddim.timesteps[i]
                z = ddim.step(unet(z, t, encoder_hidden_states=emb_c).sample, t, z).prev_sample
            t_cur = ddim.timesteps[min(n_steps, ESD["ddim_steps"] - 1)]
            e0 = frozen(z, t_cur, encoder_hidden_states=emb_null).sample
            ec = frozen(z, t_cur, encoder_hidden_states=emb_c).sample
            target = e0 - ESD["eta"] * (ec - e0)
        loss = F.mse_loss(unet(z, t_cur, encoder_hidden_states=emb_c).sample, target)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            log(f"  esd {concept} step {step} loss={loss.item():.5f}")
    unet.eval()
    for p in params:
        p.requires_grad_(False)
    del frozen, opt
    torch.cuda.empty_cache()
    return time.time() - t0


@torch.no_grad()
def eos_embedding(pipe, prompt):
    enc = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    ids = enc.input_ids[0]
    h = pipe.text_encoder(enc.input_ids.to(pipe.device))[0][0]
    return h[int((ids == pipe.tokenizer.eos_token_id).nonzero()[0].item())].float()


@torch.no_grad()
def run_uce(pipe, concepts, log):
    t0 = time.time()
    erase = [pre + e for c in concepts for e in UCE_ERASE[c] for pre in UCE["prefixes"]]
    E = torch.stack([eos_embedding(pipe, p) for p in erase], 1)
    A = torch.stack([eos_embedding(pipe, UCE["anchor"]) for _ in erase], 1)
    P = torch.stack([eos_embedding(pipe, p) for p in UCE["preserve"]], 1)
    out = {}
    for name, module in pipe.unet.named_modules():
        if "attn2" in name and (name.endswith("to_k") or name.endswith("to_v")):
            W = module.weight.data.float()
            d = E.shape[0]
            G = E @ E.T + P @ P.T + UCE["lam"] * torch.eye(d, device=E.device)
            num = (W @ A) @ E.T + (W @ P) @ P.T + UCE["lam"] * W
            out[name + ".weight"] = num @ torch.linalg.inv(G)
    log(f"  uce {concepts} {time.time() - t0:.1f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["ours", "esd_seq", "esd_merge", "uce_seq", "uce_joint"])
    ap.add_argument("--concepts", nargs="+", default=CHAIN)
    ap.add_argument("--out", default=str(OUT))
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
    cfg = dict(CFG)
    caps = retain_captions()
    layers = kv_layers()
    save_json(out / "protocol.json", dict(experiment="cross_domain", cfg=cfg, esd=ESD, uce=UCE, chain=chain, retain=caps,
                                          hashes={p: sha(ROOT / p) for p in ["dicm/experiments/cross_domain.py", "dicm/module/concept_bench.py"]},
                                          nsfw_note="nudity images are scored in memory only; no nsfw image is written to disk"))
    pipe = load_sd15_pipeline(device="cuda")
    for part in [pipe.unet, pipe.text_encoder, pipe.vae]:
        part.eval().requires_grad_(False)
    before = fingerprint(pipe.unet)
    verify_base(out, before, pipe.unet.dtype)
    bench = Bench(pipe, out, caps["test"], chain)
    log("preparing base images")
    base_summary = bench.prepare_base()
    save_json(out / "base_summary.json", base_summary)
    log("base: " + json.dumps(base_summary))
    assert fingerprint(pipe.unet) == before

    cache_path = out / "training_cache.pt"
    prompts = sorted({p for c in chain for w in ("train", "eval", "val") for p in prompts_for(c, w)} | set(caps["train"]) | set(neutral_prompts()))
    if cache_path.exists():
        raw = torch.load(cache_path, weights_only=True, map_location="cuda")
        latents, embeddings = raw["latents"], raw["embeddings"]
    else:
        log(f"caching {len(prompts)} latents")
        latents = cache_prompt_latents(pipe, prompts, cfg["cache_steps"], cfg["train_seed"])
        with torch.no_grad():
            embeddings = {p: _embed(pipe, p).detach() for p in prompts + [""]}
        torch.save(dict(latents=latents, embeddings=embeddings), cache_path)
    assert fingerprint(pipe.unet) == before

    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    acceptance = {}

    def evaluate(tag, weights, concs):
        if tag in results:
            log(f"skip {tag}")
            return
        log(f"EVAL {tag}")
        t0 = time.time()
        with weight_override(pipe.unet, weights):
            res = bench.evaluate(tag, concs)
        assert fingerprint(pipe.unet) == before
        res["seconds"] = time.time() - t0
        results[tag] = res
        save_json(results_path, results)
        log("  " + json.dumps({c: (None if v["success"] is None else round(v["success"], 2)) for c, v in res["erasure"].items()})
            + f" retain {res['retain']['dino']:.3f}/{res['retain']['clip']:.3f}")

    if "ours" in args.arms:
        lock_path = out / "retain_lock.pt"
        if lock_path.exists():
            locks = {n: t.cuda() for n, t in torch.load(lock_path, weights_only=True, map_location="cpu").items()}
        else:
            log("computing retain lock")
            locks, info = retain_lock(pipe, layers, latents, embeddings, caps["train"], cfg)
            torch.save({n: t.cpu() for n, t in locks.items()}, lock_path)
            save_json(out / "retain_lock.json", info)
        assert fingerprint(pipe.unet) == before
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
                bank, hist, acc = train_module(pipe, layers, c, latents, embeddings, caps["train"], library, locks, cfg, log, bench)
                assert fingerprint(pipe.unet) == before
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(bank, path)
                save_json(out / "history" / f"{c}.json", dict(seconds=time.time() - t0, history=hist, acceptance=acc))
                acceptance[c] = acc
                save_json(out / "acceptance.json", acceptance)
            compiled.append((c, bank))
            evaluate(f"ours/{c}", dense_weights(pipe.unet, bank), [c])
            if len(compiled) >= 2:
                members = [x for x, _ in compiled]
                evaluate("ours/" + "+".join(members), dense_weights(pipe.unet, compose([b for _, b in compiled])), members)

    del latents, embeddings
    torch.cuda.empty_cache()
    if set(args.arms) & {"esd_seq", "esd_merge", "uce_seq", "uce_joint"}:
        fp32 = load_sd15_pipeline(device="cuda", dtype=torch.float32)
        for part in [fp32.text_encoder, fp32.vae]:
            part.eval().requires_grad_(False)
        unet32 = fp32.unet.eval()
        base_state = {n: p.detach().clone().cpu() for n, p in unet32.named_parameters()}
        names = [n for n, _ in unet32.named_parameters() if "attn2" in n]
        deltas_path = out / "baseline_deltas.pt"
        store = torch.load(deltas_path, weights_only=True, map_location="cpu") if deltas_path.exists() else {}

        def reset(delta=None):
            with torch.no_grad():
                params = dict(unet32.named_parameters())
                for n, p in params.items():
                    p.data.copy_(base_state[n].to(p.device))
                for n, d in (delta or {}).items():
                    params[n].data.add_(d.to(p.device, params[n].dtype))

        def capture():
            cur = dict(unet32.named_parameters())
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
                train_esdx(fp32, c, log)
                store[f"esd/{c}"] = capture()
                torch.save(store, deltas_path)
            tag = "esd/seq_" + "_".join(chain[:i + 1])
            if tag not in store:
                log(f"TRAIN {tag}")
                reset(store["esd/seq_" + "_".join(chain[:i])] if i else None)
                train_esdx(fp32, c, log)
                store[tag] = capture()
                torch.save(store, deltas_path)
            tag = "uce/seq_" + "_".join(chain[:i + 1])
            if tag not in store:
                reset(store["uce/seq_" + "_".join(chain[:i])] if i else None)
                w = run_uce(fp32, [c], log)
                store[tag] = {n: (v.float().cpu() - base_state[n].float()).to(torch.float16) for n, v in w.items()}
                torch.save(store, deltas_path)
            tag = "uce/joint_" + "_".join(chain[:i + 1])
            if tag not in store and i:
                reset()
                w = run_uce(fp32, chain[:i + 1], log)
                store[tag] = {n: (v.float().cpu() - base_state[n].float()).to(torch.float16) for n, v in w.items()}
                torch.save(store, deltas_path)
            if i == 0:
                store["uce/joint_" + c] = store["uce/seq_" + c]
                torch.save(store, deltas_path)
        reset()
        del fp32, unet32
        torch.cuda.empty_cache()

        params = dict(pipe.unet.named_parameters())

        def apply(delta):
            return {n: (params[n].detach().float() + v.to("cuda").float()).to(params[n].dtype) for n, v in delta.items()}

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
    save_json(out / "final_status.json", dict(experiment="cross_domain", complete=True, chain=chain,
                                              fingerprint_before=before, fingerprint_after=fingerprint(pipe.unet)))
    log("COMPLETE")


if __name__ == "__main__":
    main()
