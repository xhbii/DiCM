"""Evaluate subsets for the DiCM experiments."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dicm.utils.artifacts import verify_base

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dicm.models.sd15_wrapper import load_sd15_pipeline  # noqa: E402
from dicm.module.conflict_bench import (  # noqa: E402
    Bench, configure, fill, fingerprint, retain_captions, save_json, sha, weight_override,
)
from dicm.module.sparse_delta import compose, dense_weights  # noqa: E402

from dicm.experiments.subsets import CONCEPTS8, SUBSET_SEED, SUBSET_PLAN, make_subsets

UCE = dict(lam=1.0, anchor="a photo of an object",
           erase_templates=["a photo of {a}", "{A}", "a picture of {a}"],
           preserve=["a person", "a car", "a tree", "a flower", "a bicycle", "a boat", "a sofa", "a bed",
                     "an apple", "a bottle", "a banana", "a building", "a table", "a chair", "a lamp", "a street"])




def tag_of(subset):
    return "+".join(subset)


@torch.no_grad()
def eos_embedding(pipe, prompt):
    enc = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt")
    ids = enc.input_ids[0]
    h = pipe.text_encoder(enc.input_ids.to(pipe.device))[0][0]
    eos = int((ids == pipe.tokenizer.eos_token_id).nonzero()[0].item())
    return h[eos].float()


def uce_closed_form(W, E, Vstar, P, lam):
    d = E.shape[0]
    G = E @ E.T + P @ P.T + lam * torch.eye(d, device=E.device)
    return (Vstar @ E.T + (W @ P) @ P.T + lam * W) @ torch.linalg.inv(G)


@torch.no_grad()
def uce_weights(pipe, concepts, base_state):
    """Closed-form UCE over `concepts`, solved against the pristine weights. Returns name -> new weight."""
    erase = [fill(t, c) for c in concepts for t in UCE["erase_templates"]]
    E = torch.stack([eos_embedding(pipe, p) for p in erase], 1)
    A = torch.stack([eos_embedding(pipe, UCE["anchor"]) for _ in erase], 1)
    P = torch.stack([eos_embedding(pipe, f"a photo of {p}") for p in UCE["preserve"]], 1)
    out = {}
    for mname, module in pipe.unet.named_modules():
        if "attn2" in mname and (mname.endswith("to_k") or mname.endswith("to_v")):
            name = mname + ".weight"
            W = base_state[name].to(pipe.device).float()
            out[name] = uce_closed_form(W, E, W @ A, P, UCE["lam"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["dicm", "esd_merge", "uce_merge", "uce_joint", "lora_merge"])
    ap.add_argument("--modules", help="dicm: directory holding modules/additive/<concept>.pt")
    ap.add_argument("--deltas", help="esd_merge: directory holding esd_<concept>.pt dense deltas")
    ap.add_argument("--seed-label", default="", help="recorded label of the training seed of --modules")
    ap.add_argument("--retain-only", action="store_true",
                    help="score only the 24 retain captions per configuration (DINO and CLIP). The concept "
                         "images are what make a full pass expensive, so this is about nine times faster; "
                         "erasure is left unmeasured.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.arm == "dicm" and not args.modules:
        ap.error("--arm dicm requires --modules")
    if args.arm in {"esd_merge", "lora_merge"} and not args.deltas:
        ap.error("this arm requires --deltas")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    torch.set_num_threads(4)
    configure(CONCEPTS8)
    subsets = make_subsets()
    caps = retain_captions()
    pipe = load_sd15_pipeline(device="cuda")
    for part in [pipe.unet, pipe.text_encoder, pipe.vae]:
        part.eval().requires_grad_(False)
    before = fingerprint(pipe.unet)
    verify_base(out, before, pipe.unet.dtype)
    params = dict(pipe.unet.named_parameters())
    base_state = {n: p.detach().clone() for n, p in params.items()}

    sources = {}
    if args.arm == "dicm":
        mdir = Path(args.modules)
        banks = {c: torch.load(mdir / f"{c}.pt", weights_only=True, map_location="cpu") for c in CONCEPTS8}
        sources = {c: sha(mdir / f"{c}.pt") for c in CONCEPTS8}
    elif args.arm in ("esd_merge", "lora_merge"):
        # Stacked adapters compose the same way a merged weight difference does: the sum of the per-concept
        # dense deltas. The LoRA modules were exported as (alpha/r) B A by 498 for exactly this reason.
        prefix = "esd" if args.arm == "esd_merge" else "lora"
        ddir = Path(args.deltas)
        banks = {c: torch.load(ddir / f"{prefix}_{c}.pt", weights_only=True, map_location="cpu") for c in CONCEPTS8}
        sources = {c: sha(ddir / f"{prefix}_{c}.pt") for c in CONCEPTS8}
    elif args.arm == "uce_merge":
        log("solving per-concept UCE deltas")
        banks = {}
        for c in CONCEPTS8:
            w = uce_weights(pipe, [c], base_state)
            banks[c] = {n: (v - base_state[n].float()).cpu() for n, v in w.items()}
        torch.save(banks, out / "uce_single_deltas.pt")
        sources = {c: "recomputed" for c in CONCEPTS8}
    else:
        banks = {}

    save_json(out / "protocol.json", dict(
        experiment="evaluate_subsets", rq="RQ2-A", arm=args.arm, seed_label=args.seed_label, concepts=CONCEPTS8,
        retain_only=args.retain_only,
        subset_seed=SUBSET_SEED, subset_plan=SUBSET_PLAN, subsets=[tag_of(s) for s in subsets],
        uce=UCE if args.arm.startswith("uce") else None, modules=args.modules, deltas=args.deltas,
        module_hashes=sources, retain=caps, pairs=False,
        hashes={p: sha(ROOT / p) for p in ["dicm/experiments/evaluate_subsets.py", "dicm/module/conflict_bench.py",
                                           "dicm/module/sparse_delta.py"]},
    ))

    scored = [] if args.retain_only else CONCEPTS8
    bench = Bench(pipe, out, caps["test"])
    log("preparing base retain images" if args.retain_only
        else f"preparing base images for all {len(CONCEPTS8)} concepts")
    bench.prepare_base(scored, pairs=False)
    assert fingerprint(pipe.unet) == before
    if not args.retain_only:
        base_summary = {c: dict(present=sum(r["det"] >= 0.5 for r in bench.base["single:" + c]["rows"]),
                                n=len(bench.base["single:" + c]["rows"])) for c in CONCEPTS8}
        save_json(out / "base_summary.json", base_summary)

    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    def weights_for(subset):
        if args.arm == "dicm":
            return dense_weights(pipe.unet, compose([banks[c] for c in subset]))
        if args.arm == "uce_joint":
            return {n: v.to(params[n].dtype) for n, v in uce_weights(pipe, subset, base_state).items()}
        acc = {}
        for c in subset:
            for n, v in banks[c].items():
                acc[n] = acc.get(n, 0) + v.to("cuda").float()
        return {n: (base_state[n].float() + v).to(params[n].dtype) for n, v in acc.items()}

    for i, subset in enumerate(subsets, 1):
        tag = tag_of(subset)
        if tag in results:
            log(f"skip {tag}")
            continue
        t0 = time.time()
        log(f"[{i}/{len(subsets)}] EVAL k={len(subset)} {tag}")
        w = weights_for(subset)
        with weight_override(pipe.unet, w):
            res = bench.evaluate(tag, scored, pairs=False)
        assert fingerprint(pipe.unet) == before
        res["selected"] = subset
        res["unselected"] = [c for c in CONCEPTS8 if c not in subset]
        res["seconds"] = time.time() - t0
        results[tag] = res
        save_json(results_path, results)
        sel = {c: res["single"][c]["erasure_success"] for c in subset if c in res["single"]}
        uns = {c: (None if res["single"][c]["erasure_success"] is None else round(1 - res["single"][c]["erasure_success"], 2))
               for c in res["unselected"] if c in res["single"]}
        log("  " + json.dumps(dict(selected_erasure={k: round(v, 2) for k, v in sel.items() if v is not None},
                                   unselected_still_detected=uns,
                                   retain=dict(dino=round(res["retain"]["dino"], 3), clip=round(res["retain"]["clip"], 3)),
                                   seconds=round(res["seconds"]))))
    save_json(out / "final_status.json", dict(experiment="evaluate_subsets", arm=args.arm, retain_only=args.retain_only,
                                              complete=True,
                                              fingerprint_before=before, fingerprint_after=fingerprint(pipe.unet),
                                              subsets=len(subsets)))
    log("COMPLETE")


if __name__ == "__main__":
    main()
