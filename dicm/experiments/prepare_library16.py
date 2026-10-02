import json, random, re, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import dicm.module.conflict_bench as cb
from dicm.module.library16 import BANNED_WORDS, LIBRARIES, install_templates
from dicm.models.sd15_wrapper import load_sd15_pipeline

import argparse

def main():
    ap = argparse.ArgumentParser(description="Prepare detector-filtered retain sets for the 16-object libraries.")
    ap.parse_args()
    install_templates(cb)
    OUT = ROOT / "data/library16"
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = load_sd15_pipeline(device="cuda")
    det = cb.Detector("cuda")
    data = json.loads(cb.COCO_CAPTIONS.read_text())


    def caption_pool(concepts):
        banned = re.compile(r"\b(" + "|".join(w for c in concepts for w in BANNED_WORDS[c]) + r")s?\b", re.I)
        pool, seen = [], set()
        for row in data["annotations"]:
            cap = " ".join(row["caption"].strip().split()); low = cap.lower().rstrip(".")
            if low in seen or not cap.isascii() or re.search(r"\d", cap) or banned.search(cap): continue
            n = len(cap.split())
            if n < 6 or n > 14: continue
            seen.add(low); pool.append(cap.rstrip(".").lower())
        random.Random(0).shuffle(pool)
        return pool


    for lib, concepts in LIBRARIES.items():
        path = OUT / f"retain_{lib}.json"
        if path.exists():
            print(lib, "retain exists"); continue
        t0 = time.time(); pool = caption_pool(concepts); clean, dropped = [], []
        for cap in pool:
            im = cb.generate(pipe, cap, cb.RETAIN_SEED)
            s = det.scores(im, concepts); hit = [c for c, v in s.items() if v >= 0.5]
            (dropped if hit else clean).append((cap, hit))
            if len(clean) >= 120: break
        caps = dict(test=[c for c, _ in clean[:24]], train=[c for c, _ in clean[24:120]], dropped=dropped,
                    concepts=concepts, screened=len(clean) + len(dropped))
        json.dump(caps, open(path, "w"), indent=1)
        print(f"{lib}: screened {caps['screened']} captions, dropped {len(dropped)} ({time.time()-t0:.0f}s)")
        for cap, hit in dropped[:10]: print("   dropped:", cap, hit)

    pres_path = OUT / "presence.json"
    presence = json.load(open(pres_path)) if pres_path.exists() else {}
    all_concepts = sorted({c for cs in LIBRARIES.values() for c in cs})
    for c in all_concepts:
        if c in presence: continue
        rows = []
        for t in cb.EVAL_SINGLE:
            for seed in cb.EVAL_SEEDS:
                p = cb.fill(t, c); rows.append(dict(prompt=p, seed=seed, det=det.scores(cb.generate(pipe, p, seed), [c])[c]))
        val = [det.scores(cb.generate(pipe, cb.fill(t, c), cb.VAL_SEED), [c])[c] for t in cb.VAL_SINGLE]
        presence[c] = dict(eval_present=sum(r["det"] >= 0.5 for r in rows), eval_n=len(rows),
                           val_present=sum(v >= 0.5 for v in val), val_n=len(val), rows=rows)
        json.dump(presence, open(pres_path, "w"), indent=1)
        print(f"{c:12s} eval {presence[c]['eval_present']}/{presence[c]['eval_n']}  val {presence[c]['val_present']}/{presence[c]['val_n']}", flush=True)
    print("PREP_COMPLETE")

if __name__ == "__main__":
    main()
