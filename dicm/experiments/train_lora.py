"""Train lora for the DiCM experiments."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from dicm.utils.artifacts import verify_base

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dicm.models.sd15_wrapper import load_sd15_pipeline  # noqa: E402
from dicm.module.conflict_bench import (  # noqa: E402
    CONCEPTS, DET_THRESHOLD, TRAIN_CONTEXTS, VAL_SEED, VAL_SINGLE, Bench, configure, fill, fingerprint,
    generate, kv_layers, retain_captions, save_json, sha, weight_override,
)
from dicm.module.latents import _add_noise, _embed, cache_prompt_latents  # noqa: E402

CONCEPTS8 = ["cat", "horse", "dog", "elephant", "zebra", "giraffe", "bear", "sheep"]
NEUTRAL = ["person", "car", "bicycle", "boat", "bus", "airplane", "chair", "couch", "umbrella", "clock"]
CFG = dict(rank=4, alpha=8.0, lr=1e-4, steps=200, eta=3.0, retain_weight=8.0, retain_batch=8,
           cohesion_weight=4.0, cohesion_batch=2, interaction_weight=4.0, accept_round=100, accept_max=600,
           accept_rate=0.83, subset_prob=0.5, t_min=20, t_max=980, scale=4096.0, train_seed=17, cache_steps=10)


class LoRAParam(nn.Module):
    """weight + (alpha / r) B A, with B initialised at zero so the module starts as the identity."""

    def __init__(self, weight: torch.Tensor, rank: int, alpha: float):
        super().__init__()
        out_f, in_f = weight.shape
        self.A = nn.Parameter(torch.randn(rank, in_f, device=weight.device, dtype=torch.float32) / rank**0.5)
        self.B = nn.Parameter(torch.zeros(out_f, rank, device=weight.device, dtype=torch.float32))
        self.scale = alpha / rank
        self.register_buffer("library", torch.zeros(weight.shape, device=weight.device, dtype=torch.float32))
        self.mode = "a"
        self.enabled = True

    def delta(self) -> torch.Tensor:
        return self.scale * (self.B @ self.A)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.mode == "off":
            return weight
        extra = self.delta() if self.mode == "a" else (
            self.library if self.mode == "b" else self.delta() + self.library)
        return weight + extra.to(weight.dtype)


@contextmanager
def lora_supports(unet, layers, rank, alpha):
    named = dict(unet.named_modules())
    mods, installed = {}, []
    try:
        for name in layers:
            linear = named[name.removesuffix(".weight")]
            p = LoRAParam(linear.weight, rank, alpha)
            parametrize.register_parametrization(linear, "weight", p)
            installed.append(linear)
            mods[name] = p
        yield mods
    finally:
        for linear in reversed(installed):
            parametrize.remove_parametrizations(linear, "weight", leave_parametrized=False)


def set_mode(mods, mode):
    for m in mods.values():
        m.mode = mode


@contextmanager
def disabled(mods):
    for m in mods.values():
        m.enabled = False
    try:
        yield
    finally:
        for m in mods.values():
            m.enabled = True


def set_library(mods, deltas):
    for name, m in mods.items():
        m.library.zero_()
        if deltas is not None and name in deltas:
            m.library.add_(deltas[name].to(m.library.device, torch.float32))


def export_dense(mods):
    return {n: m.delta().detach().to(torch.float16).cpu() for n, m in mods.items()}


def noisy(pipe, x0, t, gen):
    noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=gen)
    return _add_noise(pipe.scheduler, x0, noise, t)


def validation_rate(pipe, det, concept, mods):
    prev = next(iter(mods.values())).mode
    set_mode(mods, "a")
    ok = 0
    for t in VAL_SINGLE:
        im = generate(pipe, fill(t, concept), VAL_SEED)
        if det.scores(im, [concept])[concept] < DET_THRESHOLD:
            ok += 1
    set_mode(mods, prev)
    return ok / len(VAL_SINGLE)


def train_one(pipe, layers, concept, latents, embeddings, retain_train, library, cfg, arm, det, log):
    unet = pipe.unet
    idx = CONCEPTS8.index(concept)
    gen = torch.Generator(device="cuda").manual_seed(cfg["train_seed"] + idx)
    rng = random.Random(cfg["train_seed"] + 7 * idx)
    own = [fill(t, concept) for t in TRAIN_CONTEXTS]
    neutral = [fill(t, n) for n in NEUTRAL for t in TRAIN_CONTEXTS]
    lib_concepts = list(library["concepts"]) if library else []
    use_lib = bool(lib_concepts) and arm == "discipline"
    null = embeddings[""]
    history, accepted = [], None
    with lora_supports(unet, layers, cfg["rank"], cfg["alpha"]) as mods:
        if library:
            set_library(mods, library["delta"])
        params = [p for m in mods.values() for p in (m.A, m.B)]
        opt = torch.optim.Adam(params, lr=cfg["lr"])
        max_steps = cfg["accept_max"] if arm == "discipline" else cfg["steps"]
        step = 0
        while step < max_steps:
            step += 1
            t = torch.randint(cfg["t_min"], cfg["t_max"], (1,), device="cuda", generator=gen)
            p_c = rng.choice(own)
            x_c = noisy(pipe, latents[p_c], t, gen)
            rs = rng.sample(retain_train, cfg["retain_batch"])
            x_r = [(p, noisy(pipe, latents[p], t, gen)) for p in rs]
            x_n = [(p, noisy(pipe, latents[p], t, gen)) for p in rng.sample(neutral, cfg["cohesion_batch"])]
            with torch.no_grad(), disabled(mods):
                e0 = unet(x_c, t, encoder_hidden_states=null).sample.float()
                ec = unet(x_c, t, encoder_hidden_states=embeddings[p_c]).sample.float()
                target = e0 - cfg["eta"] * (ec - e0)
                t_ret = [unet(x, t, encoder_hidden_states=embeddings[p]).sample.float() for p, x in x_r]
                t_neu = [unet(x, t, encoder_hidden_states=embeddings[p]).sample.float() for p, x in x_n]
            opt.zero_grad(set_to_none=True)
            set_mode(mods, "a")
            eff = F.mse_loss(unet(x_c, t, encoder_hidden_states=embeddings[p_c]).sample.float(), target)
            (cfg["scale"] * eff).backward()
            ret_v = 0.0
            for (p, x), tt in zip(x_r, t_ret):
                l = F.mse_loss(unet(x, t, encoder_hidden_states=embeddings[p]).sample.float(), tt) / len(x_r)
                (cfg["scale"] * cfg["retain_weight"] * l).backward()
                ret_v += float(l)
            coh_v = 0.0
            if arm == "discipline":
                for (p, x), tt in zip(x_n, t_neu):
                    l = F.mse_loss(unet(x, t, encoder_hidden_states=embeddings[p]).sample.float(), tt) / len(x_n)
                    (cfg["scale"] * cfg["cohesion_weight"] * l).backward()
                    coh_v += float(l)
            int_v = 0.0
            if use_lib:
                picks = [c for c in lib_concepts if rng.random() < cfg["subset_prob"]] or [rng.choice(lib_concepts)]
                sub = {}
                for n in layers:
                    acc = torch.zeros_like(mods[n].library)
                    for c in picks:
                        acc += library["per_concept"][c][n].to(acc.device, torch.float32)
                    sub[n] = acc
                probe_p = rng.choice([p_c, rs[0], fill(rng.choice(TRAIN_CONTEXTS), rng.choice(lib_concepts))])
                x_p = noisy(pipe, latents[probe_p], t, gen)
                with torch.no_grad():
                    set_library(mods, sub); set_mode(mods, "b")
                    f_l = unet(x_p, t, encoder_hidden_states=embeddings[probe_p]).sample.float()
                    with disabled(mods):
                        f_0 = unet(x_p, t, encoder_hidden_states=embeddings[probe_p]).sample.float()
                set_mode(mods, "a")
                f_m = unet(x_p, t, encoder_hidden_states=embeddings[probe_p]).sample.float()
                set_mode(mods, "union")
                f_lm = unet(x_p, t, encoder_hidden_states=embeddings[probe_p]).sample.float()
                inter = F.mse_loss(f_lm - f_l, (f_m - f_0).detach())
                (cfg["scale"] * cfg["interaction_weight"] * inter).backward()
                int_v = float(inter)
                set_library(mods, library["delta"]); set_mode(mods, "a")
            for p in params:
                if p.grad is not None:
                    p.grad.div_(cfg["scale"])
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % 50 == 0 or step == 1:
                history.append(dict(step=step, effect=round(float(eff), 5), retain=round(ret_v, 6),
                                    cohesion=round(coh_v, 6), interaction=round(int_v, 6)))
                log(f"  {concept} step {step} effect={float(eff):.5f} retain={ret_v:.6f}")
            if arm == "discipline" and step % cfg["accept_round"] == 0:
                r = validation_rate(pipe, det, concept, mods)
                log(f"  {concept} acceptance after {step} steps: {r:.2f}")
                if accepted is None or r > accepted["rate"]:
                    accepted = dict(steps=step, rate=r, state=export_dense(mods))
                if r >= cfg["accept_rate"]:
                    break
        if arm == "discipline" and accepted is not None:
            delta = accepted["state"]
            acc = dict(steps=accepted["steps"], rate=accepted["rate"])
        else:
            delta = export_dense(mods)
            acc = dict(steps=step, rate=None)
    return delta, history, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["plain", "discipline"], required=True)
    ap.add_argument("--rank", type=int, default=CFG["rank"])
    ap.add_argument("--lr", type=float, default=CFG["lr"])
    ap.add_argument("--steps", type=int, default=CFG["steps"])
    ap.add_argument("--train-seed", type=int, default=CFG["train_seed"])
    ap.add_argument("--skip-eval", action="store_true",
                    help="train and export the modules only; the 39-subset harness (494) scores them")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    torch.set_num_threads(4)
    cfg = dict(CFG, rank=args.rank, lr=args.lr, steps=args.steps, train_seed=args.train_seed)
    configure(CONCEPTS8)
    caps = retain_captions(extra_train=72)
    layers = kv_layers()
    pipe = load_sd15_pipeline(device="cuda")
    for part in [pipe.unet, pipe.text_encoder, pipe.vae]:
        part.eval().requires_grad_(False)
    before = fingerprint(pipe.unet)
    verify_base(out, before, pipe.unet.dtype)
    save_json(out / "protocol.json", dict(experiment="train_lora", arm=args.arm, cfg=cfg, concepts=CONCEPTS8,
                                          layers=layers, retain=caps, neutral=NEUTRAL,
                                          lock="not applicable to a low-rank adapter",
                                          hashes={p: sha(ROOT / p) for p in ["dicm/experiments/train_lora.py"]}))
    prompts = ([fill(t, c) for c in CONCEPTS8 for t in TRAIN_CONTEXTS] + caps["train"]
               + [fill(t, n) for n in NEUTRAL for t in TRAIN_CONTEXTS])
    cache = out / "training_cache.pt"
    if cache.exists():
        raw = torch.load(cache, weights_only=True, map_location="cuda")
        latents, embeddings = raw["latents"], raw["embeddings"]
    else:
        log("caching latents")
        latents = cache_prompt_latents(pipe, prompts, cfg["cache_steps"], cfg["train_seed"])
        with torch.no_grad():
            embeddings = {p: _embed(pipe, p).detach() for p in prompts + [""]}
        torch.save(dict(latents=latents, embeddings=embeddings), cache)
    assert fingerprint(pipe.unet) == before

    bench = Bench(pipe, out, caps["test"])
    log("preparing base images")
    if not args.skip_eval:
        bench.prepare_base(CONCEPTS8, pairs=False)
    assert fingerprint(pipe.unet) == before
    deltas_dir = out / "deltas"; deltas_dir.mkdir(exist_ok=True)
    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    acc_path = out / "acceptance.json"
    acceptance = json.loads(acc_path.read_text()) if acc_path.exists() else {}
    per_concept = {}
    params = dict(pipe.unet.named_parameters())

    def evaluate(tag, concs, delta):
        if args.skip_eval:
            return
        if tag in results:
            log(f"skip {tag}"); return
        log(f"EVAL {tag}")
        w = {n: (params[n].detach().float() + v.to("cuda").float()).to(params[n].dtype) for n, v in delta.items()}
        with weight_override(pipe.unet, w):
            res = bench.evaluate(tag, concs, pairs=False)
        assert fingerprint(pipe.unet) == before
        results[tag] = res
        save_json(results_path, results)
        log("  " + json.dumps(dict(erase={c: round(v["erasure_success"], 2) for c, v in res["single"].items()
                                          if v["erasure_success"] is not None},
                                   retain=round(res["retain"]["dino"], 3))))

    chain = []
    for c in CONCEPTS8:
        path = deltas_dir / f"lora_{c}.pt"
        if path.exists():
            per_concept[c] = torch.load(path, weights_only=True, map_location="cpu")
        else:
            library = None
            if chain and args.arm == "discipline":
                acc = {n: torch.zeros_like(params[n], dtype=torch.float32) for n in layers}
                for x in chain:
                    for n, v in per_concept[x].items():
                        acc[n] += v.to("cuda", torch.float32)
                library = dict(concepts=list(chain), delta=acc, per_concept=per_concept)
            log(f"TRAIN {args.arm}/{c} library={chain if library else None}")
            t0 = time.time()
            delta, hist, acc_info = train_one(pipe, layers, c, latents, embeddings, caps["train"],
                                              library, cfg, args.arm, bench.det, log)
            torch.save(delta, path)
            per_concept[c] = delta
            acceptance[c] = acc_info
            save_json(out / "acceptance.json", acceptance)
            save_json(out / "history" / f"{c}.json", dict(seconds=time.time() - t0, history=hist))
            log(f"  done {c} in {time.time() - t0:.0f}s")
            assert fingerprint(pipe.unet) == before
        chain.append(c)
        evaluate(f"{args.arm}/{c}", [c], per_concept[c])
        if len(chain) >= 2:
            acc = {}
            for x in chain:
                for n, v in per_concept[x].items():
                    acc[n] = acc.get(n, 0) + v.float()
            evaluate(f"{args.arm}/" + "+".join(chain), list(chain), acc)
    save_json(out / "final_status.json", dict(experiment="train_lora", arm=args.arm, complete=True,
                                              fingerprint_before=before, fingerprint_after=fingerprint(pipe.unet)))
    log("COMPLETE")


if __name__ == "__main__":
    main()
