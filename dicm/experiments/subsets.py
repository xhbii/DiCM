"""Fixed 39-configuration evaluation plan."""
import itertools
import random

CONCEPTS8 = ["cat", "horse", "dog", "elephant", "zebra", "giraffe", "bear", "sheep"]
SUBSET_SEED = 20260921
SUBSET_PLAN = [(1, 8), (2, 10), (4, 10), (6, 10), (8, 1)]

def make_subsets() -> list[list[str]]:
    """The 39 fixed configurations, chosen before any result is seen and balanced over concepts.

    Deterministic: all combinations of the given size are shuffled with SUBSET_SEED, then taken greedily
    so that the running per-concept count stays as even as possible.
    """
    rng = random.Random(SUBSET_SEED)
    out, used = [], {c: 0 for c in CONCEPTS8}
    for k, n in SUBSET_PLAN:
        pool = [list(c) for c in itertools.combinations(CONCEPTS8, k)]
        rng.shuffle(pool)
        if k in (1, 8):
            chosen = pool if k == 8 else [[c] for c in CONCEPTS8]
        else:
            chosen = []
            for _ in range(n):
                best = min(pool, key=lambda s: (max(used[c] + (c in s) for c in CONCEPTS8),
                                                sum(used[c] for c in s), pool.index(s)))
                pool.remove(best)
                chosen.append(best)
        for s in chosen:
            for c in s:
                used[c] += 1
            out.append(s)
    assert len(out) == sum(n for _, n in SUBSET_PLAN) == 39
    return out
