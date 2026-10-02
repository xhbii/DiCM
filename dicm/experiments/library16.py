"""Library16 for the DiCM experiments."""
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
    CONCEPTS, DET_THRESHOLD, TRAIN_CONTEXTS, VAL_SEED, VAL_SINGLE, Bench, conflict_summary, configure, fill,
    fingerprint, generate, kv_layers, retain_captions, save_json, sha,
)
from dicm.module.latents import _add_noise, _embed, cache_prompt_latents  # noqa: E402
from dicm.module.conflict_bench import weight_override  # noqa: E402
import dicm.module.conflict_bench as cb  # noqa: E402
from dicm.module.library16 import LIBRARIES, NEUTRAL_DEFAULT, NEUTRAL_FAR, install_templates  # noqa: E402

install_templates(cb, wide_val="--wide-val" in sys.argv)
from dicm.module.sparse_delta import (  # noqa: E402
    compose, dense_weights, disabled, export, payload_count, project, set_mode, sparse_delta_supports,
)

OUT = ROOT / "outputs/library16"
CONCEPTS8 = ["cat", "horse", "dog", "elephant", "zebra", "giraffe", "bear", "sheep"]
NEUTRAL = list(NEUTRAL_DEFAULT)
CFG = dict(
    steps=200, lr=5e-4, eta=3.0, budget=800_000, retain_weight=1.0, retain_batch=2, hold_weight=1.0,
    cohesion_weight=1.0, cohesion_batch=2, interaction_weight=4.0,
    accept_round=100, accept_max=600, accept_rate=0.83, subset_prob=0.5, retrofit_steps=100,
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


def slice_locks(mods, locks, index, slices):
    """Forbid every coordinate outside this module's congruence class (disjoint capacity mode)."""
    out = {}
    for name, m in mods.items():
        idx = torch.arange(m.delta.numel(), device=m.delta.device).reshape(m.delta.shape)
        mine = (idx % slices) == index
        base = locks[name] if locks is not None else torch.zeros_like(mine)
        out[name] = base | (~mine)
    return out


def library_tensors(mods, payloads):
    """Dense GPU deltas for each payload, keyed like mods."""
    out = []
    for p in payloads:
        d = {}
        for name, m in mods.items():
            t = torch.zeros(m.delta.shape, device=m.delta.device, dtype=torch.float32)
            item = p[name]
            t.reshape(-1)[item["indices"].to(t.device)] = item["values"].to(t.device, torch.float32)
            d[name] = t
        out.append(d)
    return out


def set_library(mods, tensors, picks):
    with torch.no_grad():
        for name, m in mods.items():
            m.library.zero_()
            for i in picks:
                m.library.add_(tensors[i][name])


def validation_rate(pipe, det, concept, mods, cfg):
    """Detector erasure success on the configured validation prompts and seeds.

    Runs with the sparse-delta parametrization active in mode "a" (the module alone), so it works
    inside training without materializing weights.
    """
    prev = next(iter(mods.values())).mode
    set_mode(mods, "a")
    ok = 0
    for t in VAL_SINGLE:
        im = generate(pipe, fill(t, concept), VAL_SEED)
        if det.scores(im, [concept])[concept] < DET_THRESHOLD:
            ok += 1
    set_mode(mods, prev)
    return ok / len(VAL_SINGLE)


def train_module(pipe, layers, concept, latents, embeddings, retain_train, library, locks, cfg, log,
                 arm="additive", det=None, init=None, lib_payloads=None, max_steps=None,
                 budget=None, slice_index=None, slices=None, adaptive=False, budget_max=None):
    """Compile one module. `library` = dict(concepts=[...], payloads=[...]) or None.

    With cfg["accept"] the module trains in rounds of accept_round steps until the acceptance test passes
    (or accept_max is reached). With cfg["subset"] the additivity target is a random library subset per step.
    `init` seeds the delta (used by retrofit).
    """
    unet = pipe.unet
    idx = CONCEPTS.index(concept)
    gen = torch.Generator(device="cuda").manual_seed(cfg["train_seed"] + idx)
    rng = random.Random(cfg["train_seed"] + 7 * idx)
    concept_prompts = [fill(t, concept) for t in TRAIN_CONTEXTS]
    neutral_prompts = [fill(t, n) for n in NEUTRAL for t in TRAIN_CONTEXTS]
    lib_concepts = list(library["concepts"]) if library else []
    use_interaction = bool(lib_concepts)
    null = embeddings[""]
    history = []
    accepted = None
    cur_budget = budget if budget is not None else cfg["budget"]
    with sparse_delta_supports(unet, layers) as mods:
        if slices is not None:
            locks = slice_locks(mods, locks, slice_index, slices)
        tensors = library_tensors(mods, library["payloads"]) if use_interaction else []
        if init is not None:
            with torch.no_grad():
                for name, m in mods.items():
                    item = init[name]
                    m.delta.reshape(-1)[item["indices"].to(m.delta.device)] = item["values"].to(m.delta.device, torch.float32)
        set_library(mods, tensors, range(len(tensors)) if use_interaction else [])
        params = [m.delta for m in mods.values()]
        opt = torch.optim.Adam(params, lr=cfg["lr"])

        def backward(loss):
            (cfg["scale"] * loss).backward()
            return float(loss.detach())

        limit = max_steps if max_steps is not None else (cfg["accept_max"] if cfg.get("accept") else cfg["steps"])
        step = 0
        while step < limit:
            t = torch.randint(cfg["t_min"], cfg["t_max"] + 1, (1,), device="cuda", generator=gen)
            p_c = rng.choice(concept_prompts)
            x_c = noisy(pipe, latents[p_c], t, gen)
            r_prompts = rng.sample(retain_train, cfg["retain_batch"])
            x_r = [noisy(pipe, latents[p], t, gen) for p in r_prompts]
            x_n = [(p, noisy(pipe, latents[p], t, gen)) for p in rng.sample(neutral_prompts, cfg["cohesion_batch"])]
            picks = []
            x_i = []
            if use_interaction:
                n = len(tensors)
                if cfg.get("subset") and n > 1 and rng.random() < cfg["subset_prob"]:
                    size = rng.randint(1, n - 1)
                    picks = rng.sample(range(n), size)
                else:
                    picks = list(range(n))
                set_library(mods, tensors, picks)
                lc = lib_concepts[rng.choice(picks)]
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
                if use_interaction:
                    set_mode(mods, "b")
                    T["fL"] = [unet(x, t, encoder_hidden_states=embeddings[p]).sample.float() for p, x in x_i]
            opt.zero_grad(set_to_none=True)
            rec = dict(step=step, t=int(t), prompt=p_c, lib=len(picks))
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
            for p in params:
                if p.grad is not None:
                    p.grad.div_(cfg["scale"])
                    assert torch.isfinite(p.grad).all(), (concept, step)
            rec["gnorm"] = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            rec.update(project(mods, cur_budget, locks))
            history.append(rec)
            step += 1
            if step % 50 == 0 or step == 1:
                log(f"  {concept} step {step} " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items() if k != 'prompt'}))
            if cfg.get("accept") and det is not None and step % cfg["accept_round"] == 0 and step < limit:
                rate = validation_rate(pipe, det, concept, mods, cfg)
                log(f"  {concept} acceptance after {step} steps: {rate:.2f}")
                history.append(dict(step=step, acceptance=rate))
                if rate >= cfg["accept_rate"]:
                    accepted = dict(steps=step, rate=rate, budget=cur_budget)
                    break
                if adaptive and budget_max is not None and cur_budget < budget_max:
                    cur_budget = min(cur_budget * 2, budget_max)
                    log(f"  {concept} raising budget to {cur_budget}")
        if cfg.get("accept") and accepted is None and det is not None:
            rate = validation_rate(pipe, det, concept, mods, cfg)
            accepted = dict(steps=step, rate=rate, budget=cur_budget, exhausted=True)
            log(f"  {concept} acceptance at limit {step}: {rate:.2f}")
        payload = export(mods)
    return payload, history, accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="budget/lr grid on cat only, evaluate cat + retain, exit")
    ap.add_argument("--arms", nargs="+", choices=["indep", "lock", "libaware", "cohesive", "additive", "additive_rev"], default=["lock", "additive", "additive_rev"])
    ap.add_argument("--library", choices=list(LIBRARIES), required=True)
    ap.add_argument("--wide-val", action="store_true", help="12 validation prompts instead of 6 (read before parsing too)")
    ap.add_argument("--neutral", choices=["default", "far"], default="default",
                    help="cohesion neutrality set: shared default, or one disjoint from and far from this library")
    ap.add_argument("--no-pairs", action="store_true", help="skip co-occurrence scoring (k^2 images)")
    ap.add_argument("--concepts", nargs="+", default=None, help="prefix of the library to run (default: all 16)")
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
    ap.add_argument("--accept", action="store_true", help="compile in rounds until the acceptance test passes")
    ap.add_argument("--accept-rate", type=float, default=CFG["accept_rate"])
    ap.add_argument("--accept-max", type=int, default=CFG["accept_max"])
    ap.add_argument("--accept-round", type=int, default=CFG["accept_round"])
    ap.add_argument("--subset-additivity", action="store_true", help="enforce additivity against random library subsets")
    ap.add_argument("--retrofit", type=int, default=0, help="global refinement passes after the greedy chain")
    ap.add_argument("--retrofit-steps", type=int, default=CFG["retrofit_steps"])
    ap.add_argument("--capacity-mode", choices=["fixed", "adaptive", "reserve", "disjoint"], default="fixed")
    ap.add_argument("--capacity-min", type=int, default=100_000, help="adaptive: starting budget")
    ap.add_argument("--capacity-cap", type=int, default=3_200_000, help="reserve: nominal allowance; per-module minimum takes priority")
    ap.add_argument("--capacity-slices", type=int, default=8, help="disjoint: number of congruence classes")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    cfg = dict(CFG, steps=args.steps, lr=args.lr, budget=args.budget, eta=args.eta, train_seed=args.train_seed,
               retain_weight=args.retain_weight, retain_batch=args.retain_batch, cohesion_weight=args.cohesion_weight,
               interaction_weight=args.interaction_weight, accept=args.accept, accept_rate=args.accept_rate,
               accept_max=args.accept_max, accept_round=args.accept_round, subset=args.subset_additivity,
               retrofit_steps=args.retrofit_steps, retrofit=args.retrofit, capacity_mode=args.capacity_mode,
               capacity_min=args.capacity_min, capacity_cap=args.capacity_cap, capacity_slices=args.capacity_slices)
    if args.neutral == "far":
        NEUTRAL[:] = NEUTRAL_FAR[args.library]
    assert not (set(NEUTRAL) & set(LIBRARIES[args.library])), "neutrality set overlaps the library"
    if args.concepts is None:
        args.concepts = LIBRARIES[args.library]
    assert all(c in LIBRARIES[args.library] for c in args.concepts)
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
    caps = json.loads((ROOT / f"data/library16/retain_{args.library}.json").read_text())
    caps = dict(train=caps["train"], test=caps["test"])
    assert len(caps["train"]) == 96 and len(caps["test"]) == 24
    layers = kv_layers()
    save_json(out / "protocol.json", dict(
        experiment="library16", library=args.library, cfg=cfg, capacity_mode=args.capacity_mode, extra_retain=args.extra_retain, tag=args.tag, concepts=concepts, arms=args.arms, layers=layers, retain=caps, neutral=NEUTRAL, wide_val=args.wide_val, val_prompts=list(cb.VAL_SINGLE), pairs=not args.no_pairs,
        train_contexts=TRAIN_CONTEXTS, composition="sum_of_sparse_deltas",
        hashes={p: sha(ROOT / p) for p in ["dicm/experiments/library16.py", "dicm/module/library16.py", "dicm/module/conflict_bench.py", "dicm/module/sparse_delta.py"]},
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
    bench.prepare_base(concepts, pairs=not args.no_pairs)
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

    acc_path = out / "acceptance.json"
    acceptance = json.loads(acc_path.read_text()) if acc_path.exists() else {}
    capacity_used = {"total": 0}

    def capacity_args(c):
        """Per-module budget and slice for the configured capacity mode."""
        mode = args.capacity_mode
        i = concepts.index(c)
        if mode == "adaptive":
            return dict(budget=args.capacity_min, adaptive=True, budget_max=args.budget)
        if mode == "reserve":
            remaining = max(args.capacity_cap - capacity_used["total"], args.capacity_min)
            return dict(budget=max(remaining // 2, args.capacity_min))
        if mode == "disjoint":
            return dict(budget=args.budget, slice_index=i % args.capacity_slices, slices=args.capacity_slices)
        return dict(budget=args.budget)

    def train_and_store(arm, c, library, use_lock, init=None, max_steps=None, force=False):
        path = module_path(arm, c)
        if path.exists() and not force:
            log(f"reuse {arm}/{c}")
            bank = get_module(arm, c)
            capacity_used["total"] += support_count(bank)
            return bank
        log(f"TRAIN {arm}/{c} library={None if library is None else library['concepts']} lock={use_lock}")
        t0 = time.time()
        if force and path.exists():
            capacity_used["total"] -= support_count(get_module(arm, c))
        bank, hist, acc = train_module(pipe, layers, c, latents, embeddings, caps["train"], library,
                                       locks if use_lock else None, cfg, log, arm=arm, det=bench.det,
                                       init=init, max_steps=max_steps, **capacity_args(c))
        used = support_count(bank)
        capacity_used["total"] += used
        if acc:
            acc = dict(acc, coordinates=used, library_total=capacity_used["total"])
            acceptance[f"{arm}{args.tag}/{c}"] = acc
            save_json(out / "acceptance.json", acceptance)
        assert fingerprint(pipe.unet) == before
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bank, path)
        modules.pop(f"{arm}/{c}", None)
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
            res = bench.evaluate(tag, concs, save_png=False, pairs=not args.no_pairs)
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
        capacity_used["total"] = 0
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
                library = None if not chain else dict(concepts=[x for x, _ in chain], payloads=[b for _, b in chain])
                bank = train_and_store(arm, c, library, True)
                chain.append((c, bank))
                evaluate(f"{arm}/{c}", bank, [c])
                if len(chain) >= 2:
                    members = sorted([x for x, _ in chain], key=concepts.index)  # canonical order; union is order-free
                    evaluate(f"{arm}/" + "+".join(members), union_all([b for _, b in chain]), members)
                if args.stop_after and c == args.stop_after:
                    log(f"stop after {c}")
                    break
            if args.retrofit and len(chain) > 1:
                for r in range(args.retrofit):
                    log(f"RETROFIT pass {r + 1}/{args.retrofit}")
                    for i, (c, bank) in enumerate(list(chain)):
                        others = [(x, b) for j, (x, b) in enumerate(chain) if j != i]
                        library = dict(concepts=[x for x, _ in others], payloads=[b for _, b in others])
                        checkpoint = out / "refinement" / f"pass{r+1}" / "modules" / (arm + args.tag) / f"{c}.pt"
                        if checkpoint.exists():
                            new_bank = torch.load(checkpoint, weights_only=True, map_location="cpu")
                        else:
                            new_bank, hist, acc = train_module(
                                pipe, layers, c, latents, embeddings, caps["train"], library,
                                locks, cfg, log, arm=arm, det=bench.det, init=bank,
                                max_steps=args.retrofit_steps, budget=args.budget)
                            assert fingerprint(pipe.unet) == before
                            checkpoint.parent.mkdir(parents=True, exist_ok=True)
                            torch.save(new_bank, checkpoint)
                            save_json(checkpoint.with_suffix(".json"), dict(history=hist, acceptance=acc))
                        chain[i] = (c, new_bank)
                    members = sorted([x for x, _ in chain], key=concepts.index)
                    for kk in range(2, len(members) + 1):
                        sub = members[:kk]
                        evaluate(f"{arm}/r{r + 1}_" + "+".join(sub),
                                 union_all([dict(chain)[x] for x in sub]), sub)
                    for c, _ in chain:
                        evaluate(f"{arm}/r{r + 1}_{c}", dict(chain)[c], [c])
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
    save_json(out / "final_status.json", dict(experiment="library16", library=args.library, capacity_mode=args.capacity_mode, complete=True, fingerprint_before=before,
                                              fingerprint_after=fingerprint(pipe.unet), arms=args.arms, concepts=concepts))
    log("COMPLETE")


if __name__ == "__main__":
    main()
