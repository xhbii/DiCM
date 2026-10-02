"""Library16 baselines for the DiCM experiments."""
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
import dicm.module.conflict_bench as cb  # noqa: E402
from dicm.module.conflict_bench import (  # noqa: E402
    CONCEPTS, Bench, conflict_summary, configure, fill, fingerprint, save_json, sha, weight_override,
)
from dicm.module.latents import _embed  # noqa: E402
from dicm.module.library16 import LIBRARIES, UCE_PRESERVE, install_templates  # noqa: E402

install_templates(cb)

ESD = dict(steps=200, lr=1e-5, eta=3.0, ddim_steps=20, prompt="a photo of {a}", seed=0)
UCE = dict(lam=1.0, anchor="a photo of an object",
           erase_templates=["a photo of {a}", "{A}", "a picture of {a}"],
           note="Closed-form editing with the identity regularizer.")


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


def train_esdx(pipe, concept, index, log):
    """ESD-x from the CURRENT unet state (the frozen reference is that same state). Mutates the unet."""
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
    g = torch.Generator(device="cuda").manual_seed(ESD["seed"] + index)
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
        loss = F.mse_loss(unet(z, t_cur, encoder_hidden_states=emb_c).sample, target)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0 or step == 1:
            log(f"  esd {concept} step {step} loss={loss.item():.5f}")
    unet.eval()
    for p in params:
        p.requires_grad_(False)
    del frozen, opt
    torch.cuda.empty_cache()
    return time.time() - t0


@torch.no_grad()
def eos_embedding(pipe, prompt):
    """Raw EOS-position CLIP hidden state (768,), fp32: the pathway of the validated uce_delta payloads."""
    enc = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt")
    ids = enc.input_ids[0]
    h = pipe.text_encoder(enc.input_ids.to(pipe.device))[0][0]
    eos = int((ids == pipe.tokenizer.eos_token_id).nonzero()[0].item())
    return h[eos].float()


def uce_closed_form(W, E, Vstar, P, lam):
    """W' = (V* E^T + (W P) P^T + lam W)(E E^T + P P^T + lam I)^-1 (Gandikota et al. 2024)."""
    d = E.shape[0]
    G = E @ E.T + P @ P.T + lam * torch.eye(d, device=E.device)
    return (Vstar @ E.T + (W @ P) @ P.T + lam * W) @ torch.linalg.inv(G)


@torch.no_grad()
def run_uce(pipe, concepts, preserve, name, log):
    """Closed-form UCE on the CURRENT unet cross-attention K/V. Returns name -> new weight (fp32)."""
    t0 = time.time()
    erase = [fill(t, c) for c in concepts for t in UCE["erase_templates"]]
    E = torch.stack([eos_embedding(pipe, p) for p in erase], 1)
    A = torch.stack([eos_embedding(pipe, UCE["anchor"]) for _ in erase], 1)
    P = torch.stack([eos_embedding(pipe, f"a photo of {p}") for p in preserve], 1)
    out = {}
    for mname, module in pipe.unet.named_modules():
        if "attn2" in mname and (mname.endswith("to_k") or mname.endswith("to_v")):
            W = module.weight.data.float()
            out[mname + ".weight"] = uce_closed_form(W, E, W @ A, P, UCE["lam"])
    log(f"  uce {name} {time.time() - t0:.1f}s over {len(out)} tensors")
    return out


def phase_train(out, chain, preserve, log):
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

    def store(kind, tag, delta, seconds, extra=None):
        path = out / "deltas" / f"{kind}_{tag}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(delta, path)
        manifest[f"{kind}/{tag}"] = dict(path=str(path.resolve()), seconds=seconds, tensors=len(delta),
                                         params=sum(int(v.numel()) for v in delta.values()), sha256=sha(path),
                                         **(extra or {}))
        save_json(manifest_path, manifest)

    def have(kind, tag):
        return f"{kind}/{tag}" in manifest and (ROOT / manifest[f"{kind}/{tag}"]["path"]).exists()

    def get(kind, tag):
        return torch.load(ROOT / manifest[f"{kind}/{tag}"]["path"], weights_only=True, map_location="cpu")

    for i, c in enumerate(chain):
        if have("esd", c):
            continue
        log(f"TRAIN esd single {c}")
        load_delta(unet, base_state, {})
        secs = train_esdx(pipe, c, i, log)
        store("esd", c, delta_of(unet, base_state, names), secs)
    for k in range(2, len(chain) + 1):
        tag = "seq_" + "_".join(chain[:k]).replace(" ", "-")
        if have("esd", tag):
            continue
        prev = get("esd", chain[0]) if k == 2 else get("esd", "seq_" + "_".join(chain[:k - 1]).replace(" ", "-"))
        log(f"TRAIN esd {tag}")
        load_delta(unet, base_state, prev)
        secs = train_esdx(pipe, chain[k - 1], k, log)
        store("esd", tag, delta_of(unet, base_state, names), secs, dict(order=chain[:k]))
    load_delta(unet, base_state, {})
    assert fingerprint(unet) == before

    def uce_delta(weights):
        return {n: (w.float().cpu() - base_state[n].float()).to(torch.float16) for n, w in weights.items()}

    for c in chain:
        if have("uce", c):
            continue
        load_delta(unet, base_state, {})
        t0 = time.time()
        store("uce", c, uce_delta(run_uce(pipe, [c], preserve, f"single_{c}", log)), time.time() - t0)
    for k in range(2, len(chain) + 1):
        tag = "seq_" + "_".join(chain[:k]).replace(" ", "-")
        if not have("uce", tag):
            prev = get("uce", chain[0]) if k == 2 else get("uce", "seq_" + "_".join(chain[:k - 1]).replace(" ", "-"))
            load_delta(unet, base_state, prev)
            t0 = time.time()
            store("uce", tag, uce_delta(run_uce(pipe, [chain[k - 1]], preserve, tag, log)), time.time() - t0,
                  dict(order=chain[:k]))
    load_delta(unet, base_state, {})
    assert fingerprint(unet) == before
    del pipe, unet
    torch.cuda.empty_cache()
    return manifest


def phase_train_joint(out, chain, preserve, prefixes, manifest, log):
    """UCE joint solves, one per evaluated prefix (needs the pristine model each time)."""
    pipe = load_sd15_pipeline(device="cuda", dtype=torch.float32)
    unet = pipe.unet.eval()
    base_state = {n: p.detach().clone().cpu() for n, p in unet.named_parameters()}
    manifest_path = out / "train_manifest.json"
    for k in prefixes:
        if k < 2:
            continue
        tag = "joint_" + "_".join(chain[:k]).replace(" ", "-")
        if f"uce/{tag}" in manifest and (ROOT / manifest[f"uce/{tag}"]["path"]).exists():
            continue
        load_delta(unet, base_state, {})
        t0 = time.time()
        w = run_uce(pipe, chain[:k], preserve, tag, log)
        delta = {n: (v.float().cpu() - base_state[n].float()).to(torch.float16) for n, v in w.items()}
        path = out / "deltas" / f"uce_{tag}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(delta, path)
        manifest[f"uce/{tag}"] = dict(path=str(path.resolve()), seconds=time.time() - t0,
                                      tensors=len(delta), params=sum(int(v.numel()) for v in delta.values()),
                                      sha256=sha(path), order=chain[:k])
        save_json(manifest_path, manifest)
    del pipe, unet
    torch.cuda.empty_cache()
    return manifest


def merged(deltas):
    out = {}
    for d in deltas:
        for n, v in d.items():
            out[n] = out.get(n, 0) + v.float()
    return out


def phase_eval(out, chain, caps, prefixes, manifest, log):
    pipe = load_sd15_pipeline(device="cuda")
    for part in [pipe.unet, pipe.text_encoder, pipe.vae]:
        part.eval().requires_grad_(False)
    before = fingerprint(pipe.unet)
    verify_base(out, before, pipe.unet.dtype)
    bench = Bench(pipe, out, caps["test"])
    log("preparing base images")
    bench.prepare_base(chain, pairs=False)
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
            res = bench.evaluate(tag, concs, pairs=False)
        assert fingerprint(pipe.unet) == before
        res["seconds"] = time.time() - t0
        res["delta_norm"] = float(sum((v.float() ** 2).sum() for v in delta.values()) ** 0.5) if delta else 0.0
        results[tag] = res
        save_json(results_path, results)
        log("  " + json.dumps(dict(single={c: v["erasure_success"] for c, v in res["single"].items()},
                                   retain=dict(dino=round(res["retain"]["dino"], 3), clip=round(res["retain"]["clip"], 3)),
                                   seconds=round(res["seconds"]))))

    evaluate("base", {}, chain)
    for kind in ["esd", "uce"]:
        for c in chain:
            evaluate(f"{kind}/{c}", get(kind, c), [c])
        for k in prefixes:
            if k < 2:
                continue
            concs = chain[:k]
            slug = "_".join(concs).replace(" ", "-")
            evaluate(f"{kind}/merge_{slug}", merged([get(kind, c) for c in concs]), concs)
            evaluate(f"{kind}/seq_{slug}", get(kind, f"seq_{slug}"), concs)
            if kind == "uce":
                evaluate(f"uce/joint_{slug}", get("uce", f"joint_{slug}"), concs)
    summary = {}
    for kind in ["esd", "uce"]:
        sub = {k: v for k, v in results.items() if k.startswith(kind + "/")}
        summary[kind] = conflict_summary(sub, {c: f"{kind}/{c}" for c in chain})
    save_json(out / "summary.json", summary)
    save_json(out / "final_status.json", dict(experiment="library16_baselines", complete=True, fingerprint_before=before,
                                              fingerprint_after=fingerprint(pipe.unet)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", choices=list(LIBRARIES), required=True)
    ap.add_argument("--phase", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--prefixes", default="1,2,4,8,12,16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()  # delta paths are stored relative to ROOT, so this must be absolute
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    torch.set_num_threads(4)
    chain = LIBRARIES[args.library]
    preserve = UCE_PRESERVE[args.library]
    assert not (set(chain) & {p.split(" ", 1)[1] for p in preserve}), "UCE preserve list overlaps the library"
    prefixes = [int(x) for x in args.prefixes.split(",")]
    caps = json.loads((ROOT / f"data/library16/retain_{args.library}.json").read_text())
    caps = dict(train=caps["train"], test=caps["test"])
    configure(chain)
    save_json(out / "protocol.json", dict(
        experiment="library16_baselines", library=args.library, esd=ESD, uce=dict(UCE, preserve=preserve), concepts=chain,
        prefixes=prefixes, pairs=False, retain=caps,
        hashes={p: sha(ROOT / p) for p in ["dicm/experiments/library16_baselines.py", "dicm/module/conflict_bench.py",
                                           "dicm/module/library16.py"]},
    ))
    manifest_path = out / "train_manifest.json"
    if args.phase in ("train", "all"):
        manifest = phase_train(out, chain, preserve, log)
        manifest = phase_train_joint(out, chain, preserve, prefixes, manifest, log)
    else:
        manifest = json.loads(manifest_path.read_text())
    if args.phase in ("eval", "all"):
        phase_eval(out, chain, caps, prefixes, manifest, log)
    log("COMPLETE")


if __name__ == "__main__":
    main()
