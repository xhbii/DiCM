"""Aggregate per-configuration results without replacing missing measurements."""
import json
import statistics
from pathlib import Path
from dicm.experiments.subsets import make_subsets


def mean(values):
    xs = [v for v in values if v is not None]
    return statistics.mean(xs) if xs else None


def summarize(directory):
    directory = Path(directory)
    protocol = json.loads((directory / 'protocol.json').read_text())
    results = json.loads((directory / 'results.json').read_text())
    expected = {'+'.join(s) for s in make_subsets()}
    if set(results) != expected or set(protocol.get('subsets', [])) != expected:
        raise ValueError(f'{directory}: expected the complete fixed 39-configuration plan')
    retain_only = protocol.get('retain_only', False)
    singles = {s[0]: results[s[0]].get('single', {}).get(s[0], {}).get('erasure_success')
               for s in make_subsets() if len(s) == 1}
    rows = []
    for tag, result in results.items():
        selected, unselected = result['selected'], result['unselected']
        if tag != '+'.join(selected) or set(unselected) != set(singles) - set(selected):
            raise ValueError(f'Invalid subset metadata: {tag}')
        def rate(c):
            if not retain_only and c not in result['single']:
                raise ValueError(f'Missing concept score: {tag}/{c}')
            value = result.get('single', {}).get(c, {}).get('erasure_success')
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f'Invalid erasure rate: {tag}/{c}')
            return value
        erasure = [rate(c) for c in selected]
        keep = [None if rate(c) is None else 1-rate(c) for c in unselected]
        known = [v for v in erasure if v is not None]
        changes = [rate(c)-singles[c] for c in selected if rate(c) is not None and singles[c] is not None]
        conflict = any(-v >= 0.2-1e-12 for v in changes) if len(changes) == len(selected) else None
        for metric in ('dino', 'clip'):
            if metric not in result['retain']:
                raise ValueError(f'Missing retain {metric}: {tag}')
        rows.append(dict(tag=tag, k=len(selected), erasure=mean(erasure),
                         min_member=min(known) if known else None,
                         unselected_retention=mean(keep), dino=result['retain']['dino'],
                         clip=result['retain']['clip'], conflict=conflict,
                         measured_members=len(known), selected_members=len(selected)))
    aggregate = []
    for k in (1, 2, 4, 6, 8):
        group = [row for row in rows if row['k'] == k]
        minima = [row['min_member'] for row in group if row['min_member'] is not None]
        aggregate.append(dict(k=k, n=len(group), erasure=mean(row['erasure'] for row in group),
                              min_member=min(minima) if minima else None,
                              unselected_retention=mean(row['unselected_retention'] for row in group),
                              dino=mean(row['dino'] for row in group), clip=mean(row['clip'] for row in group),
                              conflicts=sum(row['conflict'] is True for row in group),
                              conflict_evaluable=sum(row['conflict'] is not None for row in group)))
    return dict(run=directory.name, arm=protocol['arm'], seed=protocol.get('seed_label'),
                retain_only=retain_only, by_k=aggregate, configurations=rows)
