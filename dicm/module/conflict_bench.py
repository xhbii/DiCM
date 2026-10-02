"""Conflict bench: detector-based erasure success of single modules and their compositions.

Shared by scripts 473 (deletion modules v2), 474 (ESD-x / UCE baselines) and 475 (report).

Design decisions (2026-09-15):
- Concepts are COCO detector classes so erasure success is measured by Faster R-CNN, not CLIP margins.
- Evaluation prompts are held out from every training prompt; retain prompts are COCO captions that
  mention none of the concepts, split into a training half and a held-out half.
- "Conflict" of a composition is the drop in a concept's erasure success relative to its own single module.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]

CONCEPTS = ["cat", "horse", "dog", "elephant"]
BANNED_ALL = {
    "cat": ["cat", "kitten", "kitty", "feline"],
    "horse": ["horse", "pony", "stallion", "mare", "equestrian", "foal"],
    "dog": ["dog", "puppy", "canine", "pup", "hound"],
    "elephant": ["elephant"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "bear": ["bear", "grizzly", "panda"],
    "sheep": ["sheep", "lamb", "ram", "ewe"],
    "cow": ["cow", "cattle", "bull", "calf", "ox", "oxen"],
    "bird": ["bird", "parrot", "pigeon", "seagull", "gull", "duck", "goose", "geese", "owl", "eagle"],
}
BANNED = {c: BANNED_ALL[c] for c in CONCEPTS}


def configure(concepts: list[str]) -> None:
    """Switch the bench to another concept set (in place, so `from ... import CONCEPTS` bindings follow)."""
    CONCEPTS[:] = list(concepts)
    BANNED.clear()
    BANNED.update({c: BANNED_ALL[c] for c in concepts})
TRAIN_CONTEXTS = [
    "a photograph of {a} standing on a wooden floor, fully visible",
    "a photo of {a} in a sunny garden",
    "{A} standing in a grassy field, daytime",
    "a close-up photograph of {a} looking at the camera",
    "{A} next to a brick wall",
    "a documentary photograph of {a} on a farm",
]
EVAL_SINGLE = [
    "a photograph of {a} on a beach at sunset",
    "{A} in a snowy forest",
    "{A} standing in a city street",
    "a professional photo of {a} in a studio",
    "{A} beside a lake with mountains behind",
    "{A} in a living room",
    "{A} on a dirt road in the countryside",
    "{A} under a large oak tree",
    "a photograph of {a} in a parking lot",
    "{A} in front of a red barn",
    "{A} on a stone bridge",
    "{A} in a meadow full of flowers",
]
VAL_SINGLE = [
    "{A} in a backyard on a sunny afternoon",
    "a photograph of {a} near a wooden shed",
    "{A} on a gravel path",
    "a photo of {a} beside a stone wall",
    "{A} in front of a hedge",
    "a photograph of {a} on dry grass",
]
# Six further validation contexts and a second seed. The six-template, one-seed gate accepted a seed-29
# sibling-cohesion dog module at 100 steps whose held-out erasure was 0.17: a solution that only held in the
# validation contexts. Widening the gate costs four times as many images per acceptance round.
VAL_WIDE = VAL_SINGLE + [
    "{A} on a paved driveway",
    "a photograph of {a} at the edge of a wood",
    "{A} beside a parked car",
    "a photo of {a} under an overcast sky",
    "{A} on a patch of bare earth",
    "a photograph of {a} next to a picket fence",
]
VAL_SEED = 3001
VAL_SEEDS_WIDE = [3001, 3002]
EVAL_PAIR = [
    "a photograph of {a} and {b} together in a park",
    "{A} standing next to {b} in a field",
    "a photo of {a} and {b} on a farm, both fully visible",
    "{A} and {b} in front of a wooden fence",
    "{A} beside {b} near a river",
    "a photograph of two animals, {a} on the left and {b} on the right",
]
EVAL_SEEDS = [1001, 1002]
RETAIN_SEED = 2001
DET_THRESHOLD = 0.5
GEN = dict(steps=20, guidance=7.5, size=512)
KV_LAYERS_FILE = Path(__file__).resolve().parents[1] / "configs/kv_layers_sd15.json"
COCO_CAPTIONS = Path(os.environ.get("DICM_COCO_CAPTIONS", str(ROOT / "data/captions_train2017.json")))


def article(c: str) -> str:
    return ("an " if c[0] in "aeiou" else "a ") + c


def fill(template: str, a: str, b: str | None = None) -> str:
    out = template.replace("{a}", article(a)).replace("{A}", article(a).capitalize())
    if b is not None:
        out = out.replace("{b}", article(b))
    return out


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, data) -> None:
    from dicm.utils.artifacts import write_json
    write_json(path, data)


def kv_layers() -> list[str]:
    return json.loads(KV_LAYERS_FILE.read_text())["kv"]


def retain_captions(n_train: int = 24, n_test: int = 24, seed: int = 0, extra_train: int = 0) -> dict[str, list[str]]:
    """COCO captions with 6-14 words, ascii, no digits, mentioning none of the concepts."""
    data = json.loads(COCO_CAPTIONS.read_text())
    banned = re.compile(r"\b(" + "|".join(w for ws in BANNED.values() for w in ws) + r")s?\b", re.I)
    pool = []
    seen = set()
    for row in data["annotations"]:
        cap = " ".join(row["caption"].strip().split())
        low = cap.lower().rstrip(".")
        if low in seen or not cap.isascii() or re.search(r"\d", cap) or banned.search(cap):
            continue
        n = len(cap.split())
        if n < 6 or n > 14:
            continue
        seen.add(low)
        pool.append(cap.rstrip(".").lower())
    rng = random.Random(seed)
    rng.shuffle(pool)
    # test set is always pool[n_train:n_train+n_test]; extra training captions come after it so the test set never moves
    return dict(train=pool[:n_train] + pool[n_train + n_test:n_train + n_test + extra_train], test=pool[n_train:n_train + n_test])


@torch.no_grad()
def generate(pipe, prompt: str, seed: int) -> Image.Image:
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(prompt, num_inference_steps=GEN["steps"], guidance_scale=GEN["guidance"],
                width=GEN["size"], height=GEN["size"], generator=gen).images[0]


class Detector:
    def __init__(self, device: str = "cuda"):
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights).eval().to(device)
        self.transform = weights.transforms()
        self.categories = weights.meta["categories"]
        self.device = device

    @torch.inference_mode()
    def scores(self, image: Image.Image, labels: list[str]) -> dict[str, float]:
        pred = self.model([self.transform(image).to(self.device)])[0]
        out = {}
        for label in labels:
            s = pred["scores"][pred["labels"] == self.categories.index(label)]
            out[label] = float(s.max()) if len(s) else 0.0
        return out


@contextmanager
def weight_override(unet, weights: dict[str, torch.Tensor]):
    """Temporarily load a partial state dict (name -> tensor), restore bit-exact on exit."""
    params = dict(unet.named_parameters())
    saved = {}
    try:
        with torch.no_grad():
            for name, value in weights.items():
                saved[name] = params[name].data.clone()
                params[name].data.copy_(value.to(params[name].dtype))
        yield
    finally:
        with torch.no_grad():
            for name, value in saved.items():
                params[name].data.copy_(value)


def fingerprint(net) -> str:
    digest = hashlib.sha256()
    for name, weight in sorted(net.state_dict().items()):
        digest.update(name.encode())
        digest.update(weight.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sheet(path: Path, images: list[Image.Image], labels: list[str], cols: int = 6, size: int = 192) -> None:
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * size, rows * (size + 16)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (im, lab) in enumerate(zip(images, labels)):
        x, y = (i % cols) * size, (i // cols) * (size + 16)
        canvas.paste(im.resize((size, size)), (x, y + 16))
        draw.text((x + 2, y + 2), lab[:40], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=85)


class Bench:
    """Generates and scores one model state; base images are cached once and reused."""

    def __init__(self, pipe, out: Path, retain_test: list[str], device: str = "cuda"):
        from dicm.evaluation.metrics.clipscore import CLIPScorer
        from dicm.evaluation.metrics.dinov2 import DINOv2Similarity

        self.pipe, self.out, self.retain_test = pipe, out, retain_test
        self.det = Detector(device)
        self.clip = CLIPScorer(device=device)
        self.dino = DINOv2Similarity(device=device)
        self.base: dict[str, dict] = {}

    def _single_requests(self, concept):
        return [(fill(t, concept), s) for t in EVAL_SINGLE for s in EVAL_SEEDS]

    def _pair_requests(self, a, b):
        return [(fill(t, a, b), s) for t in EVAL_PAIR for s in EVAL_SEEDS]

    def base_pack(self, concept):
        key = "single:" + concept
        if key not in self.base:
            rows, images = [], []
            for prompt, seed in self._single_requests(concept):
                im = generate(self.pipe, prompt, seed)
                rows.append(dict(prompt=prompt, seed=seed, det=self.det.scores(im, [concept])[concept]))
                images.append(im)
            self.base[key] = dict(rows=rows, images=images)
            sheet(self.out / "review/base" / f"single_{concept}.jpg", images, [f"{r['det']:.2f} {r['prompt']}" for r in rows])
        return self.base[key]

    def base_pair(self, a, b):
        key = f"pair:{a}:{b}"
        if key not in self.base:
            rows, images = [], []
            for prompt, seed in self._pair_requests(a, b):
                im = generate(self.pipe, prompt, seed)
                rows.append(dict(prompt=prompt, seed=seed, det=self.det.scores(im, [a, b])))
                images.append(im)
            self.base[key] = dict(rows=rows, images=images)
            sheet(self.out / "review/base" / f"pair_{a}_{b}.jpg", images, [f"{r['det'][a]:.2f}/{r['det'][b]:.2f}" for r in rows])
        return self.base[key]

    def base_retain(self):
        if "retain" not in self.base:
            rows, images = [], []
            for cap in self.retain_test:
                im = generate(self.pipe, cap, RETAIN_SEED)
                rows.append(dict(prompt=cap, seed=RETAIN_SEED, clip=self.clip.score(im, cap)))
                images.append(im)
            self.base["retain"] = dict(rows=rows, images=images)
            sheet(self.out / "review/base/retain.jpg", images, [r["prompt"] for r in rows])
        return self.base["retain"]

    def prepare_base(self, concepts: list[str], pairs: bool = True) -> None:
        """Generate every base image BEFORE any weight override is active (single, pairs, retain).

        ``pairs=False`` skips the co-occurrence images, which grow as k^2 and cost more than everything
        else once the library passes ~10 concepts.
        """
        for c in concepts:
            self.base_pack(c)
        if pairs:
            for i, a in enumerate(concepts):
                for b in concepts[i + 1:]:
                    self.base_pair(a, b)
        self.base_retain()
        self.prepared = True

    def evaluate(self, tag: str, concepts: list[str], save_png: bool = False, pairs: bool = True) -> dict:
        """Score the CURRENT pipe state for the given erased concepts (single + all pairs + retain).

        ``pairs=False`` scores singles and retain only; ``result["pair"]`` is then empty.
        """
        needed = ["single:" + c for c in concepts] + ["retain"]
        if pairs:
            needed += [f"pair:{a}:{b}" for i, a in enumerate(concepts) for b in concepts[i + 1:]]
        missing = [k for k in needed if k not in self.base]
        if missing:
            raise RuntimeError(f"base images missing (call prepare_base outside any override): {missing}")
        result = dict(tag=tag, concepts=list(concepts), single={}, pair={}, retain={})
        for c in concepts:
            base = self.base_pack(c)
            rows, images = [], []
            for (prompt, seed), b in zip(self._single_requests(c), base["rows"]):
                im = generate(self.pipe, prompt, seed)
                d = self.det.scores(im, [c])[c]
                rows.append(dict(prompt=prompt, seed=seed, det=d, base_det=b["det"], valid=b["det"] >= DET_THRESHOLD,
                                 erased=d < DET_THRESHOLD))
                images.append(im)
                if save_png:
                    p = self.out / "images" / tag / f"single_{c}_{seed}_{hashlib.sha256(prompt.encode()).hexdigest()[:8]}.png"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    im.save(p)
            valid = [r for r in rows if r["valid"]]
            result["single"][c] = dict(
                n_valid=len(valid), n=len(rows),
                erasure_success=(sum(r["erased"] for r in valid) / len(valid)) if valid else None,
                mean_det=sum(r["det"] for r in rows) / len(rows),
                rows=rows,
            )
            sheet(self.out / "review" / tag / f"single_{c}.jpg", images, [f"{r['det']:.2f} {r['prompt']}" for r in rows])
        for i, a in enumerate(concepts if pairs else []):
            for b in concepts[i + 1:]:
                base = self.base_pair(a, b)
                rows, images = [], []
                for (prompt, seed), bb in zip(self._pair_requests(a, b), base["rows"]):
                    im = generate(self.pipe, prompt, seed)
                    d = self.det.scores(im, [a, b])
                    valid = bb["det"][a] >= DET_THRESHOLD and bb["det"][b] >= DET_THRESHOLD
                    rows.append(dict(prompt=prompt, seed=seed, det=d, base_det=bb["det"], valid=valid,
                                     both_absent=d[a] < DET_THRESHOLD and d[b] < DET_THRESHOLD,
                                     a_absent=d[a] < DET_THRESHOLD, b_absent=d[b] < DET_THRESHOLD))
                    images.append(im)
                    if save_png:
                        p = self.out / "images" / tag / f"pair_{a}_{b}_{seed}_{hashlib.sha256(prompt.encode()).hexdigest()[:8]}.png"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        im.save(p)
                valid = [r for r in rows if r["valid"]]
                result["pair"][f"{a}+{b}"] = dict(
                    n_valid=len(valid), n=len(rows),
                    both_absent=(sum(r["both_absent"] for r in valid) / len(valid)) if valid else None,
                    a_absent=(sum(r["a_absent"] for r in valid) / len(valid)) if valid else None,
                    b_absent=(sum(r["b_absent"] for r in valid) / len(valid)) if valid else None,
                    rows=rows,
                )
                sheet(self.out / "review" / tag / f"pair_{a}_{b}.jpg", images, [f"{r['det'][a]:.2f}/{r['det'][b]:.2f}" for r in rows])
        base = self.base_retain()
        rows, images = [], []
        for cap, b, bim in zip(self.retain_test, base["rows"], base["images"]):
            im = generate(self.pipe, cap, RETAIN_SEED)
            rows.append(dict(prompt=cap, clip=self.clip.score(im, cap), base_clip=b["clip"], dino=self.dino.similarity(bim, im)))
            images.append(im)
        result["retain"] = dict(
            dino=sum(r["dino"] for r in rows) / len(rows),
            clip=sum(r["clip"] for r in rows) / len(rows),
            base_clip=sum(r["base_clip"] for r in rows) / len(rows),
            rows=rows,
        )
        sheet(self.out / "review" / tag / "retain.jpg", images, [f"{r['dino']:.2f} {r['prompt']}" for r in rows])
        return result


def conflict_summary(results: dict[str, dict], singles: dict[str, str]) -> list[dict]:
    """For every composition tag: per-concept erasure drop vs that concept's own single module.

    ``singles`` maps concept -> tag of its single-module result within ``results``.
    """
    rows = []
    for tag, res in results.items():
        if len(res["concepts"]) < 2:
            continue
        drops = {}
        for c in res["concepts"]:
            own = results[singles[c]]["single"][c]["erasure_success"]
            comp = res["single"][c]["erasure_success"]
            drops[c] = None if own is None or comp is None else own - comp
        rows.append(dict(
            tag=tag, k=len(res["concepts"]),
            erasure=[res["single"][c]["erasure_success"] for c in res["concepts"]],
            drop=drops,
            worst_drop=max((d for d in drops.values() if d is not None), default=None),
            both_absent={k: v["both_absent"] for k, v in res["pair"].items()},
            retain_dino=res["retain"]["dino"],
            retain_clip=res["retain"]["clip"],
        ))
    return rows
