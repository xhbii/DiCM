#!/usr/bin/env python3
"""Export RQ1/RQ2/RQ4 subset results to JSON, CSV, and Markdown."""
import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dicm.evaluation.report import summarize
from dicm.utils.artifacts import write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    reports = [summarize(path) for path in a.runs]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'summary.json', reports)
    flat = [dict(run=r['run'], **row) for r in reports for row in r['by_k']]
    with (out / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    lines = ['| Run | k | n | Erasure | Worst member | Unselected retained | DINO | CLIP | Conflicts / evaluable |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):
        return '—' if v is None else f'{v:.3f}'
    for row in flat:
        vals = [row['run'], str(row['k']), str(row['n'])]
        vals += [fmt(row[key]) for key in ('erasure', 'min_member', 'unselected_retention', 'dino', 'clip')]
        vals += [f"{row['conflicts']} / {row['conflict_evaluable']}"]
        lines.append('| ' + ' | '.join(vals) + ' |')
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
