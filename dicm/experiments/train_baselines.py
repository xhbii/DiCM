"""Train baselines for the DiCM experiments."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

from dicm.utils.artifacts import verify_base

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dicm.models.sd15_wrapper import load_sd15_pipeline  # noqa: E402
from dicm.module.conflict_bench import (  # noqa: E402
    CONCEPTS, Bench, conflict_summary, configure, fill, fingerprint, retain_captions, save_json, sha, weight_override,
)
from dicm.module.latents import _embed  # noqa: E402

OUT = ROOT / "outputs/train_baselines"
ESD = dict(steps=200, lr=1e-5, eta=3.0, ddim_steps=20, prompt="a photo of {a}", seed=0)
UCE = dict(lam=1.0, anchor="a photo of an object",
           erase_templates=["a photo of {a}", "{A}", "a picture of {a}"],
           preserve=["a person", "a car", "a tree", "a flower", "a bicycle", "a boat", "a bird", "a sofa", "a bed", "an apple",
                     "a bottle", "a banana", "a building", "a table", "a zebra", "a giraffe"],
           note="Closed-form editing with the identity regularizer.")

CHAIN = ["cat", "horse", "dog", "elephant", "zebra", "giraffe", "bear", "sheep"]


def pairs(concepts):
    return [(a, b) for i, a in enumerate(concepts) for b in concepts[i + 1:]]


# ---------------------------------------------------------------- phase 1: training (fp32)
def attn2_names(unet):
    return [n for n, _ in unet.named_parameters() if "attn2" in n]


def delta_of(unet, base_state, names):
    cur = dict(unet.named_parameters())
    out = {}
    for n in names:
        d = cur[n].detach().float().cpu() - base_state[n].float()
        if bool((d != 0).any()):
            out[n] = d.to(torch.float16)
    return out


def load_delta(unet, base_state, delta):
    with torch.no_grad():
        params = dict(unet.named_parameters())
        for n, p in params.items():
            if n in base_state:
                p.data.copy_(base_state[n].to(p.device))
        for n, d in delta.items():
            params[n].data.add_(d.to(params[n].device, params[n].dtype))


def train_esdx(pipe, concept, log):
    """ESD-x from the CURRENT unet state (frozen reference = current state). Returns nothing; unet mutated."""
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
        emb_c = _embed(pipe, fill(ESD["prompt"], concept)).detach()
        emb_null = _embed(pipe, "").detach()
    g = torch.Generator(device="cuda").manual_seed(ESD["seed"] + CONCEPTS.index(concept))
    unet.train()
    t0 = time.time()
    for step in range(1, ESD["steps"] + 1):
        n_steps = int(torch.randint(1, ESD["ddim_steps"], (1,), generator=g, device="cuda"))
        z = torch.randn(1, 4, 64, 64, generator=g, device="cuda")
        with torch.no_grad():
            for i in range(n_steps):
                t = ddim.timesteps[i]
                eps = unet(z, t, encoder_hidden_states=emb_c).sample
                z = ddim.step(eps, t, z).prev_sample
            t_cur = ddim.timesteps[min(n_steps, ESD["ddim_steps"] - 1)]
            e0 = frozen(z, t_cur, encoder_hidden_states=emb_null).sample
            ec = frozen(z, t_cur, encoder_hidden_states=emb_c).sample
            target = e0 - ESD["eta"] * (ec - e0)
        pred = unet(z, t_cur, encoder_hidden_states=emb_c).sample
        loss = F.mse_loss(pred, target)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == 1:
            log(f"  esd {concept} step {step} loss={loss.item():.5f}")
    unet.eval()
    for p in params:
        p.requires_grad_(False)
    del frozen, opt
    torch.cuda.empty_cache()
    return time.time() - t0


@torch.no_grad()
def eos_embedding(pipe, prompt):
    """Raw EOS-position CLIP hidden state (768,), fp32. Same pathway as the validated uce_delta payloads."""
    enc = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt")
    ids = enc.input_ids[0]
    h = pipe.text_encoder(enc.input_ids.to(pipe.device))[0][0]
    eos = int((ids == pipe.tokenizer.eos_token_id).nonzero()[0].item())
    return h[eos].float()


def uce_closed_form(W, E, Vstar, P, lam):
    """W' = (V* E^T + (W P) P^T + lam W)(E E^T + P P^T + lam I)^-1 (Gandikota et al. 2024, with identity term)."""
    d = E.shape[0]
    G = E @ E.T + P @ P.T + lam * torch.eye(d, device=E.device)
    num = Vstar @ E.T + (W @ P) @ P.T + lam * W
    return num @ torch.linalg.inv(G)


@torch.no_grad()
def run_uce(pipe, concepts, name, log):
    """Closed-form UCE on the CURRENT unet K/V weights (all 32 attn2 to_k/to_v). Returns name->new weight (fp32)."""
    t0 = time.time()
    erase = [fill(t, c) for c in concepts for t in UCE["erase_templates"]]
    anchors = [UCE["anchor"] for _ in erase]
    E = torch.stack([eos_embedding(pipe, p) for p in erase], 1)
    A = torch.stack([eos_embedding(pipe, p) for p in anchors], 1)
    P = torch.stack([eos_embedding(pipe, f"a photo of {p}") for p in UCE["preserve"]], 1)
    out = {}
    for mname, module in pipe.unet.named_modules():
        if "attn2" in mname and (mname.endswith("to_k") or mname.endswith("to_v")):
            W = module.weight.data.float()
            out[mname + ".weight"] = uce_closed_form(W, E, W @ A, P, UCE["lam"])
    log(f"  uce {name} {time.time() - t0:.1f}s {len(out)} tensors erase={erase}")
    return out


def phase_train(out, log):
    pipe = load_sd15_pipeline(device="cuda", dtype=torch.float32)
    for part in [pipe.text_encoder, pipe.vae]:
        part.eval().requires_grad_(False)
    unet = pipe.unet.eval()
    base_state = {n: p.detach().clone().cpu() for n, p in unet.named_parameters()}
    names = attn2_names(unet)
    before = fingerprint(unet)
    verify_base(out, before, unet.dtype)
    manifest_path = out / "train_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    save_json(manifest_path, manifest)

    def store(kind, tag, delta, seconds, extra=None):
        path = out / "deltas" / f"{kind}_{tag}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(delta, path)
        manifest[f"{kind}/{tag}"] = dict(path=str(path.resolve()), seconds=seconds, tensors=len(delta),
                                          params=sum(int(v.numel()) for v in delta.values()), sha256=sha(path), **(extra or {}))
        save_json(manifest_path, manifest)

    def have(kind, tag):
        return f"{kind}/{tag}" in manifest and (ROOT / manifest[f"{kind}/{tag}"]["path"]).exists()

    def get(kind, tag):
        return torch.load(ROOT / manifest[f"{kind}/{tag}"]["path"], weights_only=True, map_location="cpu")

    # ---- ESD-x singles
    for c in CONCEPTS:
        if have("esd", c):
            continue
        log(f"TRAIN esd single {c}")
        load_delta(unet, base_state, {})
        secs = train_esdx(pipe, c, log)
        store("esd", c, delta_of(unet, base_state, names), secs)
    # ---- ESD-x sequential prefixes, starting from the first singleton.
    prev = CHAIN[0]
    for k in range(2, len(CHAIN) + 1):
        tag = "seq_" + "_".join(CHAIN[:k])
        if not have("esd", tag):
            log(f"TRAIN esd {tag}")
            load_delta(unet, base_state, get("esd", prev))
            secs = train_esdx(pipe, CHAIN[k - 1], log)
            store("esd", tag, delta_of(unet, base_state, names), secs, dict(order=CHAIN[:k]))
        prev = tag
    load_delta(unet, base_state, {})
    assert fingerprint(unet) == before

    # ---- UCE
    def uce_delta(weights):
        return {n: (w.float().cpu() - base_state[n].float()).to(torch.float16) for n, w in weights.items()}

    for c in CONCEPTS:
        if have("uce", c):
            continue
        load_delta(unet, base_state, {})
        t0 = time.time()
        store("uce", c, uce_delta(run_uce(pipe, [c], f"single_{c}", log)), time.time() - t0)
    for k in range(2, len(CHAIN) + 1):
        tag = "joint_" + "_".join(CHAIN[:k])
        if not have("uce", tag):
            load_delta(unet, base_state, {})
            t0 = time.time()
            store("uce", tag, uce_delta(run_uce(pipe, CHAIN[:k], tag, log)), time.time() - t0)
        tag = "seq_" + "_".join(CHAIN[:k])
        if not have("uce", tag):
            prev = CHAIN[0] if k == 2 else "seq_" + "_".join(CHAIN[:k - 1])
            load_delta(unet, base_state, get("uce", prev))
            t0 = time.time()
            store("uce", tag, uce_delta(run_uce(pipe, [CHAIN[k - 1]], tag, log)), time.time() - t0, dict(order=CHAIN[:k]))
    load_delta(unet, base_state, {})
    assert fingerprint(unet) == before
    del pipe, unet
    torch.cuda.empty_cache()
    return manifest


# ---------------------------------------------------------------- phase 2: evaluation (fp16 bench)
def merged(deltas):
    out = {}
    for d in deltas:
        for n, v in d.items():
            out[n] = out.get(n, 0) + v.float()
    return out


def phase_eval(out, manifest, log):
    pipe = load_sd15_pipeline(device="cuda")
    for part in [pipe.unet, pipe.text_encoder, pipe.vae]:
        part.eval().requires_grad_(False)
    before = fingerprint(pipe.unet)
    verify_base(out, before, pipe.unet.dtype)
    caps = retain_captions()
    bench = Bench(pipe, out, caps["test"])
    log("preparing base images")
    bench.prepare_base(CONCEPTS)
    assert fingerprint(pipe.unet) == before
    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    params = dict(pipe.unet.named_parameters())

    def get(kind, tag):
        return torch.load(ROOT / manifest[f"{kind}/{tag}"]["path"], weights_only=True, map_location="cpu")

    def evaluate(tag, delta, concs):
        if tag in results:
            log(f"skip eval {tag}")
            return
        log(f"EVAL {tag}")
        t0 = time.time()
        weights = {n: (params[n].detach().float() + v.to("cuda").float()).to(params[n].dtype) for n, v in delta.items()}
        with weight_override(pipe.unet, weights):
            res = bench.evaluate(tag, concs)
        assert fingerprint(pipe.unet) == before
        res["seconds"] = time.time() - t0
        res["delta_params"] = sum(int(v.numel()) for v in delta.values())
        res["delta_norm"] = float(sum((v.float() ** 2).sum() for v in delta.values()) ** 0.5) if delta else 0.0
        results[tag] = res
        save_json(results_path, results)
        log("  " + json.dumps(dict(single={c: v["erasure_success"] for c, v in res["single"].items()},
                                   pair={k: v["both_absent"] for k, v in res["pair"].items()},
                                   retain=dict(dino=round(res["retain"]["dino"], 3), clip=round(res["retain"]["clip"], 3)),
                                   seconds=round(res["seconds"]))))

    evaluate("base", {}, CONCEPTS)
    for kind in ["esd", "uce"]:
        for c in CONCEPTS:
            evaluate(f"{kind}/{c}", get(kind, c), [c])
        for k in range(2, len(CHAIN) + 1):
            concs = CHAIN[:k]
            evaluate(f"{kind}/merge_" + "_".join(concs), merged([get(kind, c) for c in concs]), concs)
            evaluate(f"{kind}/seq_" + "_".join(concs), get(kind, "seq_" + "_".join(concs)), concs)
            if kind == "uce":
                evaluate("uce/joint_" + "_".join(concs), get("uce", "joint_" + "_".join(concs)), concs)
    summary = {}
    for kind in ["esd", "uce"]:
        sub = {k: v for k, v in results.items() if k.startswith(kind + "/")}
        summary[kind] = conflict_summary(sub, {c: f"{kind}/{c}" for c in CONCEPTS})
    save_json(out / "summary.json", summary)
    save_json(out / "final_status.json", dict(experiment="train_baselines", complete=True, fingerprint_before=before,
                                              fingerprint_after=fingerprint(pipe.unet)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    torch.set_num_threads(4)
    configure(CHAIN)
    save_json(out / "protocol.json", dict(
        experiment="train_baselines", esd=ESD, uce=UCE, concepts=CONCEPTS, chain=CHAIN,
        hashes={p: sha(ROOT / p) for p in ["dicm/experiments/train_baselines.py", "dicm/module/conflict_bench.py"]},
    ))
    manifest_path = out / "train_manifest.json"
    if args.phase in ("train", "all"):
        manifest = phase_train(out, log)
    else:
        manifest = json.loads(manifest_path.read_text())
    if args.phase in ("eval", "all"):
        phase_eval(out, manifest, log)
    log("COMPLETE")


if __name__ == "__main__":
    main()
