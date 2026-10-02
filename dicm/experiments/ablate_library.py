"""Eight-concept objective and preservation-strength ablations."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dicm.models.sd15_wrapper import load_sd15_pipeline  # noqa: E402
from dicm.module.backbone_partition import lock_from_scores  # noqa: E402
from dicm.module.conflict_bench import (  # noqa: E402
    CONCEPTS, TRAIN_CONTEXTS, Bench, conflict_summary, configure, fill, fingerprint, kv_layers, retain_captions, save_json, sha,
)
from dicm.module.conflict_bench import weight_override  # noqa: E402
from dicm.module.latents import _add_noise, _embed, cache_prompt_latents  # noqa: E402
from dicm.module.sparse_delta import (  # noqa: E402
    compose, dense_weights, disabled, export, install_library, payload_count, project, set_mode, sparse_delta_supports,
)

OUT = ROOT / "outputs/ablate_library"
CONCEPTS8 = ["cat", "horse", "dog", "elephant", "zebra", "giraffe", "bear", "sheep"]
NEUTRAL = ["person", "car", "bicycle", "boat", "bus", "airplane", "chair", "couch", "umbrella", "clock"]
CFG = dict(
    steps=200, lr=5e-4, eta=3.0, budget=800_000, retain_weight=1.0, retain_batch=2, hold_weight=1.0,
    cohesion_weight=1.0, cohesion_batch=2, interaction_weight=4.0,
    t_min=20, t_max=980, scale=4096.0, lock_fraction=0.5, lock_draws=16, train_seed=17, cache_steps=10,
)


def noisy(pipe, x0, t, gen):
    noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=gen)
    return _add_noise(pipe.scheduler, x0, noise, t)


def mse(a, b):
    return torch.nn.functional.mse_loss(a.float(), b.float())


def support_count(payload):
    return payload_count(payload)


def union_all(payloads):
    return compose(list(payloads))


def retain_lock(pipe, layers, latents, embeddings, retain_train, cfg):
    """|d(pred . v)/dW| accumulated over retain latents, random unit directions v, fp32, loss-scaled."""
    unet = pipe.unet
    params = {n: p for n, p in unet.named_parameters() if n in layers}
    for p in params.values():
        p.requires_grad_(True)
    acc = {n: torch.zeros(p.shape, device=p.device, dtype=torch.float32) for n, p in params.items()}
    gen = torch.Generator(device="cuda").manual_seed(cfg["train_seed"] + 99)
    for i in range(cfg["lock_draws"]):
        prompt = retain_train[i % len(retain_train)]
        t = torch.randint(cfg["t_min"], cfg["t_max"] + 1, (1,), device="cuda", generator=gen)
        xt = noisy(pipe, latents[prompt], t, gen)
        pred = unet(xt, t, encoder_hidden_states=embeddings[prompt]).sample
        v = torch.randn(pred.shape, device=pred.device, dtype=torch.float32, generator=gen)
        v = v / v.norm()
        (cfg["scale"] * (pred.float() * v).sum()).backward()
        for n, p in params.items():
            assert torch.isfinite(p.grad).all(), n
            acc[n] += p.grad.abs().float() / cfg["scale"]
            p.grad = None
    for p in params.values():
        p.requires_grad_(False)
    locks = lock_from_scores(acc, cfg["lock_fraction"])
    return locks, dict(locked=sum(int(l.sum()) for l in locks.values()), total=sum(l.numel() for l in locks.values()))


def train_module(pipe, layers, concept, latents, embeddings, retain_train, library, locks, cfg, log, arm="libaware"):
    unet = pipe.unet
    idx = CONCEPTS.index(concept)
    gen = torch.Generator(device="cuda").manual_seed(cfg["train_seed"] + idx)
    rng = random.Random(cfg["train_seed"] + 7 * idx)
    concept_prompts = [fill(t, concept) for t in TRAIN_CONTEXTS]
    neutral_prompts = [fill(t, n) for n in NEUTRAL for t in TRAIN_CONTEXTS]
    new_style = arm in ("cohesive", "additive", "additive_rev")
    use_interaction = arm in ("additive", "additive_rev") and library is not None
    null = embeddings[""]
    history = []
    with sparse_delta_supports(unet, layers) as mods:
        install_library(mods, None if library is None else library["union"])
        params = [m.delta for m in mods.values()]
        opt = torch.optim.Adam(params, lr=cfg["lr"])

        def backward(loss):
            (cfg["scale"] * loss).backward()
            return float(loss.detach())

        for step in range(cfg["steps"]):
            t = torch.randint(cfg["t_min"], cfg["t_max"] + 1, (1,), device="cuda", generator=gen)
            p_c = rng.choice(concept_prompts)
            x_c = noisy(pipe, latents[p_c], t, gen)
            r_prompts = rng.sample(retain_train, cfg["retain_batch"])
            x_r = [noisy(pipe, latents[p], t, gen) for p in r_prompts]
            hold = None
            if library is not None and not new_style:
                lc = rng.choice(library["concepts"])
                p_l = rng.choice([fill(tp, lc) for tp in TRAIN_CONTEXTS])
                hold = (p_l, noisy(pipe, latents[p_l], t, gen))
            x_n = []
            if new_style:
                for p in rng.sample(neutral_prompts, cfg["cohesion_batch"]):
                    x_n.append((p, noisy(pipe, latents[p], t, gen)))
            x_i = []
            if use_interaction:
                lc = rng.choice(library["concepts"])
                p_l = rng.choice([fill(tp, lc) for tp in TRAIN_CONTEXTS])
                x_i = [(p_c, x_c), (r_prompts[0], x_r[0]), (p_l, noisy(pipe, latents[p_l], t, gen))]
            with torch.no_grad():
                with disabled(mods):
                    e0 = unet(x_c, t, encoder_hidden_states=null).sample.float()
                    ec = unet(x_c, t, encoder_hidden_states=embeddings[p_c]).sample.float()
                    T = dict(
                        effect=e0 - cfg["eta"] * (ec - e0),
                        retain=[unet(x, t, encoder_hidden_states=embeddings[p]).sample.float() for x, p in zip(x_r, r_prompts)],
                        neutral=[unet(x, t, encoder_hidden_states=embeddings[p]).sample.float() for p, x in x_n],
                        f0=[unet(x, t, encoder_hidden_states=embeddings[p]).sample.float() for p, x in x_i],
                    )
                if hold is not None:
                    set_mode(mods, "b")
                    T["hold"] = unet(hold[1], t, encoder_hidden_states=embeddings[hold[0]]).sample.float()
                if use_interaction:
                    set_mode(mods, "b")
                    T["fL"] = [unet(x, t, encoder_hidden_states=embeddings[p]).sample.float() for p, x in x_i]
            opt.zero_grad(set_to_none=True)
            rec = dict(step=step, t=int(t), prompt=p_c)
            set_mode(mods, "a")
            pred = unet(x_c, t, encoder_hidden_states=embeddings[p_c]).sample.float()
            rec["effect"] = backward(mse(pred, T["effect"]))
            del pred
            rec["retain"] = 0.0
            for x, p, tr in zip(x_r, r_prompts, T["retain"]):
                pred = unet(x, t, encoder_hidden_states=embeddings[p]).sample.float()
                rec["retain"] += backward(cfg["retain_weight"] * mse(pred, tr) / cfg["retain_batch"])
                del pred
            if new_style:
                rec["cohesion"] = 0.0
                for (p, x), tn in zip(x_n, T["neutral"]):
                    pred = unet(x, t, encoder_hidden_states=embeddings[p]).sample.float()
                    rec["cohesion"] += backward(cfg["cohesion_weight"] * mse(pred, tn) / cfg["cohesion_batch"])
                    del pred
            if use_interaction:
                rec["interaction"] = 0.0
                for (p, x), f0, fL in zip(x_i, T["f0"], T["fL"]):
                    set_mode(mods, "a")
                    fM = unet(x, t, encoder_hidden_states=embeddings[p]).sample.float()
                    set_mode(mods, "union")
                    fLM = unet(x, t, encoder_hidden_states=embeddings[p]).sample.float()
                    inter = (fLM - fL) - (fM - f0)
                    rec["interaction"] += backward(cfg["interaction_weight"] * (inter ** 2).mean() / len(x_i))
                    del fM, fLM, inter
                set_mode(mods, "a")
            if library is not None and not new_style:
                set_mode(mods, "union")
                rec["retain_union"] = 0.0
                for x, p, tr in zip(x_r, r_prompts, T["retain"]):
                    pred = unet(x, t, encoder_hidden_states=embeddings[p]).sample.float()
                    rec["retain_union"] += backward(cfg["retain_weight"] * mse(pred, tr) / cfg["retain_batch"])
                    del pred
                pred = unet(hold[1], t, encoder_hidden_states=embeddings[hold[0]]).sample.float()
                rec["hold"] = backward(cfg["hold_weight"] * mse(pred, T["hold"]))
                del pred
            for p in params:
                if p.grad is not None:
                    p.grad.div_(cfg["scale"])
                    assert torch.isfinite(p.grad).all(), (concept, step)
            rec["gnorm"] = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            rec.update(project(mods, cfg["budget"], locks))
            history.append(rec)
            if step % 25 == 0 or step == cfg["steps"] - 1:
                log(f"  {concept} step {step} " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items() if k != 'prompt'}))
        payload = export(mods)
    return payload, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="budget/lr grid on cat only, evaluate cat + retain, exit")
    ap.add_argument("--arms", nargs="+", choices=["indep", "lock", "libaware", "cohesive", "additive", "additive_rev"], default=["lock", "additive", "additive_rev"])
    ap.add_argument("--concepts", nargs="+", default=CONCEPTS8)
    ap.add_argument("--steps", type=int, default=CFG["steps"])
    ap.add_argument("--lr", type=float, default=CFG["lr"])
    ap.add_argument("--budget", type=int, default=CFG["budget"])
    ap.add_argument("--eta", type=float, default=CFG["eta"])
    ap.add_argument("--train-seed", type=int, default=CFG["train_seed"])
    ap.add_argument("--retain-weight", type=float, default=CFG["retain_weight"])
    ap.add_argument("--retain-batch", type=int, default=CFG["retain_batch"])
    ap.add_argument("--cohesion-weight", type=float, default=CFG["cohesion_weight"])
    ap.add_argument("--interaction-weight", type=float, default=CFG["interaction_weight"])
    ap.add_argument("--extra-retain", type=int, default=0, help="extra training captions beyond the first 24 (test set unchanged)")
    ap.add_argument("--tag", default="", help="suffix appended to arm names in module/result keys")
    ap.add_argument("--stop-after", default=None, help="chain arms: stop after compiling and evaluating this concept")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    cfg = dict(CFG, steps=args.steps, lr=args.lr, budget=args.budget, eta=args.eta, train_seed=args.train_seed,
               retain_weight=args.retain_weight, retain_batch=args.retain_batch, cohesion_weight=args.cohesion_weight,
               interaction_weight=args.interaction_weight)
    configure(args.concepts)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    concepts = args.concepts
    logf = open(out / "run.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    torch.set_num_threads(4)
    caps = retain_captions(extra_train=args.extra_retain)
    layers = kv_layers()
    save_json(out / "protocol.json", dict(
        experiment="ablate_library", cfg=cfg, extra_retain=args.extra_retain, tag=args.tag, concepts=concepts, arms=args.arms, layers=layers, retain=caps, neutral=NEUTRAL,
        train_contexts=TRAIN_CONTEXTS, composition="sum_of_sparse_deltas",
        hashes={p: sha(ROOT / p) for p in ["dicm/experiments/ablate_library.py", "dicm/module/conflict_bench.py", "dicm/module/sparse_delta.py"]},
    ))
    pipe = load_sd15_pipeline(device="cuda")
    for part in [pipe.unet, pipe.text_encoder, pipe.vae]:
        part.eval().requires_grad_(False)
    before = fingerprint(pipe.unet)
    verify_base(out, before, pipe.unet.dtype)
    log(f"fingerprint {before}")

    cache_path = out / ("training_cache_x%d.pt" % args.extra_retain if args.extra_retain else "training_cache.pt")
    prompts = [fill(t, c) for c in CONCEPTS for t in TRAIN_CONTEXTS] + caps["train"] + [fill(t, n) for n in NEUTRAL for t in TRAIN_CONTEXTS]
    if cache_path.exists():
        raw = torch.load(cache_path, weights_only=True, map_location="cuda")
        latents, embeddings = raw["latents"], raw["embeddings"]
    else:
        log("caching latents")
        latents = cache_prompt_latents(pipe, prompts, cfg["cache_steps"], cfg["train_seed"])
        with torch.no_grad():
            embeddings = {p: _embed(pipe, p).detach() for p in prompts + [""]}
        torch.save(dict(latents=latents, embeddings=embeddings), cache_path)
    assert fingerprint(pipe.unet) == before

    lock_path = out / "retain_lock.pt"
    locks = None
    if set(args.arms) & {"lock", "libaware", "cohesive", "additive", "additive_rev"} or args.pilot:
        if lock_path.exists():
            locks = {n: t.cuda() for n, t in torch.load(lock_path, weights_only=True, map_location="cpu").items()}
        else:
            log("computing retain lock")
            locks, info = retain_lock(pipe, layers, latents, embeddings, caps["train"], cfg)
            torch.save({n: t.cpu() for n, t in locks.items()}, lock_path)
            save_json(out / "retain_lock.json", info)
            log(f"lock {info}")
        assert fingerprint(pipe.unet) == before

    bench = Bench(pipe, out, caps["test"])
    log("preparing base images")
    bench.prepare_base(concepts)
    assert fingerprint(pipe.unet) == before
    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    modules: dict[str, dict] = {}

    def module_path(arm, c):
        return out / "modules" / (arm + args.tag) / f"{c}.pt"

    def get_module(arm, c):
        key = f"{arm}/{c}"
        if key not in modules:
            modules[key] = torch.load(module_path(arm, c), weights_only=True, map_location="cpu")
        return modules[key]

    def train_and_store(arm, c, library, use_lock):
        path = module_path(arm, c)
        if path.exists():
            log(f"reuse {arm}/{c}")
            return get_module(arm, c)
        log(f"TRAIN {arm}/{c} library={None if library is None else library['concepts']} lock={use_lock}")
        t0 = time.time()
        bank, hist = train_module(pipe, layers, c, latents, embeddings, caps["train"], library, locks if use_lock else None, cfg, log, arm=arm)
        assert fingerprint(pipe.unet) == before
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bank, path)
        save_json(out / "history" / (arm + args.tag) / f"{c}.json", dict(seconds=time.time() - t0, deleted=support_count(bank), history=hist))
        log(f"  done {arm}/{c} deleted={support_count(bank)} {time.time() - t0:.0f}s")
        return get_module(arm, c)

    def evaluate(tag, bank, concs):
        if args.tag:
            arm_, rest = tag.split("/", 1)
            tag = f"{arm_}{args.tag}/{rest}"
        if tag in results:
            log(f"skip eval {tag}")
            return results[tag]
        log(f"EVAL {tag}")
        t0 = time.time()
        with weight_override(pipe.unet, dense_weights(pipe.unet, bank)):
            res = bench.evaluate(tag, concs, save_png=False)
        assert fingerprint(pipe.unet) == before
        res["seconds"] = time.time() - t0
        res["deleted"] = support_count(bank)
        results[tag] = res
        save_json(results_path, results)
        log("  " + json.dumps(dict(single={c: v["erasure_success"] for c, v in res["single"].items()},
                                   pair={k: v["both_absent"] for k, v in res["pair"].items()},
                                   retain=dict(dino=round(res["retain"]["dino"], 3), clip=round(res["retain"]["clip"], 3)),
                                   deleted=res["deleted"], seconds=round(res["seconds"]))))
        return res

    if args.pilot:
        for budget, lr in [(200_000, 5e-4), (200_000, 2e-3), (800_000, 5e-4), (800_000, 2e-3)]:
            cfg.update(budget=budget, lr=lr)
            arm = f"pilot_b{budget // 1000}k_lr{lr:g}"
            bank = train_and_store(arm, "cat", None, False)
            evaluate(f"{arm}/cat", bank, ["cat"])
        log("PILOT_COMPLETE")
        return

    singles = {}
    for arm in args.arms:
        if arm in ("indep", "lock", "cohesive"):
            use_lock = arm != "indep"
            banks = {c: train_and_store(arm, c, None, use_lock) for c in concepts}
            for c in concepts:
                evaluate(f"{arm}/{c}", banks[c], [c])
                singles[f"{arm}/{c}"] = f"{arm}/{c}"
            for i, a in enumerate(concepts):
                for b in concepts[i + 1:]:
                    evaluate(f"{arm}/{a}+{b}", union_all([banks[a], banks[b]]), [a, b])
            if len(concepts) >= 3:
                evaluate(f"{arm}/" + "+".join(concepts[:3]), union_all([banks[c] for c in concepts[:3]]), concepts[:3])
            for k in range(4, len(concepts) + 1):
                evaluate(f"{arm}/" + "+".join(concepts[:k]), union_all([banks[c] for c in concepts[:k]]), concepts[:k])
        elif arm in ("libaware", "additive", "additive_rev"):
            chain = []
            order = list(reversed(concepts)) if arm == "additive_rev" else list(concepts)
            for c in order:
                library = None if not chain else dict(concepts=[x for x, _ in chain], union=union_all([b for _, b in chain]))
                bank = train_and_store(arm, c, library, True)
                chain.append((c, bank))
                evaluate(f"{arm}/{c}", bank, [c])
                if len(chain) >= 2:
                    members = sorted([x for x, _ in chain], key=concepts.index)  # canonical order; union is order-free
                    evaluate(f"{arm}/" + "+".join(members), union_all([b for _, b in chain]), members)
                if args.stop_after and c == args.stop_after:
                    log(f"stop after {c}")
                    break
            if arm == "additive_rev":
                # pairs in canonical order for coupling comparison with the forward chain
                banks = dict(chain)
                for i, a in enumerate(concepts):
                    for b in concepts[i + 1:]:
                        evaluate(f"{arm}/{a}+{b}", union_all([banks[a], banks[b]]), [a, b])
    # conflict summary per arm, singles = same arm's single module
    summary = {}
    for arm in args.arms:
        sub = {k: v for k, v in results.items() if k.startswith(arm + "/")}
        summary[arm] = conflict_summary(sub, {c: f"{arm}/{c}" for c in concepts})
    save_json(out / "summary.json", summary)
    save_json(out / "final_status.json", dict(experiment="ablate_library", complete=True, fingerprint_before=before,
                                              fingerprint_after=fingerprint(pipe.unet), arms=args.arms, concepts=concepts))
    log("COMPLETE")


if __name__ == "__main__":
    main()
