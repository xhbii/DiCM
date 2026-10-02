"""Measure modularity for the DiCM experiments."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from dicm.utils.artifacts import verify_base

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dicm.models.sd15_wrapper import load_sd15_pipeline  # noqa: E402
from dicm.module.conflict_bench import CONCEPTS, EVAL_SINGLE, configure, fill, fingerprint, kv_layers, retain_captions, save_json, weight_override  # noqa: E402
from dicm.module.latents import _add_noise, _embed, cache_prompt_latents  # noqa: E402
from dicm.module.sparse_delta import compose, dense_weights  # noqa: E402

OUT = ROOT / "outputs/measure_modularity"
TIMESTEPS = [150, 400, 650, 900]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["indep", "lock", "libaware"])
    ap.add_argument("--prompts-per-concept", type=int, default=6)
    ap.add_argument("--retain", type=int, default=12)
    ap.add_argument("--concepts", nargs="+", default=None)
    ap.add_argument("--out", default=str(OUT))
    return ap.parse_args()


class Probe:
    """Fixed bank of (x_t, t, emb) inputs grouped by class; returns stacked predictions for a weight state."""

    def __init__(self, pipe, latents, embeddings, groups, seed=476):
        self.pipe = pipe
        self.items = []  # (group, x_t, t, emb)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        for group, prompts in groups.items():
            for p in prompts:
                x0 = latents[p]
                for t in TIMESTEPS:
                    tt = torch.tensor([t], device="cuda")
                    self.items.append((group, _add_noise(pipe.scheduler, x0, torch.randn(x0.shape, device="cuda", dtype=x0.dtype, generator=gen), tt), tt, embeddings[p]))
        self.groups = list(groups)

    @torch.no_grad()
    def run(self, weights):
        out = {g: [] for g in self.groups}
        with weight_override(self.pipe.unet, weights):
            for g, x, t, e in self.items:
                out[g].append(self.pipe.unet(x, t, encoder_hidden_states=e).sample.float().cpu())
        return {g: torch.stack(v) for g, v in out.items()}


def energy(d):
    return float((d ** 2).mean(dim=(1, 2, 3, 4)).mean()) if d.ndim == 5 else float((d ** 2).mean())


def main():
    args = parse()
    if args.concepts:
        configure(args.concepts)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if len(CONCEPTS) < 2:
        raise ValueError("Modularity measurements require at least two concepts")
    import hashlib
    sources = {}
    for run in args.runs:
        for arm in args.arms:
            for concept in CONCEPTS:
                path = ROOT / run / "modules" / arm / f"{concept}.pt"
                if not path.is_file():
                    raise FileNotFoundError(path)
                sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(out / "protocol.json", dict(concepts=list(CONCEPTS), runs=args.runs,
        arms=args.arms, prompts_per_concept=args.prompts_per_concept,
        retain=args.retain, timesteps=TIMESTEPS, modules=sources))
    torch.set_num_threads(4)
    pipe = load_sd15_pipeline(device="cuda")
    for part in [pipe.unet, pipe.text_encoder, pipe.vae]:
        part.eval().requires_grad_(False)
    before = fingerprint(pipe.unet)
    verify_base(out, before, pipe.unet.dtype)
    caps = retain_captions()["test"][: args.retain]
    groups = {c: [fill(t, c) for t in EVAL_SINGLE[: args.prompts_per_concept]] for c in CONCEPTS}
    groups["retain"] = caps
    prompts = [p for ps in groups.values() for p in ps]
    cache = out / "probe_cache.pt"
    if cache.exists():
        raw = torch.load(cache, weights_only=True, map_location="cuda")
        latents, embeddings = raw["latents"], raw["embeddings"]
    else:
        latents = cache_prompt_latents(pipe, prompts, 10, 4760)
        with torch.no_grad():
            embeddings = {p: _embed(pipe, p).detach() for p in prompts}
        torch.save(dict(latents=latents, embeddings=embeddings), cache)
    probe = Probe(pipe, latents, embeddings, groups)
    f0 = probe.run({})
    assert fingerprint(pipe.unet) == before
    params = dict(pipe.unet.named_parameters())
    results = {}
    for run in args.runs:
        rdir = ROOT / run
        for arm in args.arms:
            if not all((rdir / "modules" / arm / f"{c}.pt").exists() for c in CONCEPTS):
                print("skip", run, arm, flush=True)
                continue
            mods = {c: torch.load(rdir / "modules" / arm / f"{c}.pt", weights_only=True, map_location="cpu") for c in CONCEPTS}
            key = f"{Path(run).name}/{arm}"
            print("==", key, flush=True)
            single = {c: probe.run(dense_weights(pipe.unet, mods[c])) for c in CONCEPTS}
            eff = {c: {g: single[c][g] - f0[g] for g in probe.groups} for c in CONCEPTS}
            # cohesion
            coh = {}
            for c in CONCEPTS:
                own = energy(eff[c][c])
                others = {g: energy(eff[c][g]) for g in probe.groups if g != c}
                coh[c] = dict(own=own, other_concepts=sum(v for g, v in others.items() if g != "retain") / (len(CONCEPTS) - 1),
                              retain=others["retain"], ratio=own / max(sum(others.values()) / len(others), 1e-12))
            # pairwise coupling
            pair = {}
            for a, b in itertools.combinations(CONCEPTS, 2):
                fab = probe.run(dense_weights(pipe.unet, compose([mods[a], mods[b]])))
                row = {}
                for g in probe.groups:
                    inter = fab[g] - single[a][g] - single[b][g] + f0[g]
                    row[g] = energy(inter) / max(energy(eff[a][g]) + energy(eff[b][g]), 1e-12)
                row["concepts"] = sum(row[g] for g in CONCEPTS) / len(CONCEPTS)
                pair[f"{a}+{b}"] = row
            # self-consistency inside the full library (full) and in the compile-order library (prefix)
            full = probe.run(dense_weights(pipe.unet, compose([mods[c] for c in CONCEPTS])))
            selfc = {}
            for c in CONCEPTS:
                others = [mods[o] for o in CONCEPTS if o != c]
                fl = probe.run(dense_weights(pipe.unet, compose(others)))
                num = {g: energy((full[g] - fl[g]) - eff[c][g]) for g in probe.groups}
                den = {g: max(energy(eff[c][g]), 1e-12) for g in probe.groups}
                selfc[c] = dict(own=num[c] / den[c], retain=num["retain"] / den["retain"],
                                all=sum(num.values()) / sum(den.values()))
            # full collateral additivity on retain
            add4 = energy(full["retain"] - f0["retain"] - sum(eff[c]["retain"] for c in CONCEPTS)) / max(sum(energy(eff[c]["retain"]) for c in CONCEPTS), 1e-12)
            results[key] = dict(cohesion=coh, coupling=pair, self_consistency=selfc, library_retain_interaction=add4,
                                summary=dict(
                                    cohesion_ratio_mean=sum(v["ratio"] for v in coh.values()) / len(coh),
                                    coupling_concepts_mean=sum(v["concepts"] for v in pair.values()) / len(pair),
                                    coupling_retain_mean=sum(v["retain"] for v in pair.values()) / len(pair),
                                    self_consistency_own_mean=sum(v["own"] for v in selfc.values()) / len(selfc),
                                    library_retain_interaction=add4))
            print(json.dumps(results[key]["summary"], indent=1), flush=True)
            assert fingerprint(pipe.unet) == before
            save_json(out / "results.json", results)
    # table
    lines = ["# Cohesion / coupling metrics", "",
             "cohesion = own-concept effect energy / mean(other concepts, retain) effect energy (higher = module does one thing).",
             "coupling = interaction energy of a pair / sum of single effect energies, on concept inputs and on retain inputs (0 = additive).",
             "self-consistency = how much a module's effect inside the full library differs from its effect alone, relative to its own effect (0 = same).", "",
             "| run/arm | cohesion ratio | coupling (concepts) | coupling (retain) | self-consistency (own) | full retain interaction |", "|---|---|---|---|---|---|"]
    for k, r in results.items():
        s = r["summary"]
        lines.append(f"| {k} | {s['cohesion_ratio_mean']:.2f} | {s['coupling_concepts_mean']:.3f} | {s['coupling_retain_mean']:.3f} | {s['self_consistency_own_mean']:.3f} | {s['library_retain_interaction']:.3f} |")
    lines += ["", "## Per concept cohesion (own / other concepts / retain effect energy, ratio)", "", "| run/arm | concept | own | other concepts | retain | ratio |", "|---|---|---|---|---|---|"]
    for k, r in results.items():
        for c, v in r["cohesion"].items():
            lines.append(f"| {k} | {c} | {v['own']:.5f} | {v['other_concepts']:.5f} | {v['retain']:.5f} | {v['ratio']:.2f} |")
    lines += ["", "## Per pair coupling", "", "| run/arm | pair | concepts | retain |", "|---|---|---|---|"]
    for k, r in results.items():
        for p, v in r["coupling"].items():
            lines.append(f"| {k} | {p} | {v['concepts']:.3f} | {v['retain']:.3f} |")
    lines += ["", "## Self-consistency in the full library", "", "| run/arm | concept | own | retain | all |", "|---|---|---|---|---|"]
    for k, r in results.items():
        for c, v in r["self_consistency"].items():
            lines.append(f"| {k} | {c} | {v['own']:.3f} | {v['retain']:.3f} | {v['all']:.3f} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
