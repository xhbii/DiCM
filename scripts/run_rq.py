#!/usr/bin/env python3
"""Execute the checked-in RQ plans, or print commands without loading models."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def commands(rq, seed, stage='all', names=None, extensions=False):
    plan = json.loads((ROOT / 'experiments' / f'{rq}.json').read_text())
    known = {j['name'] for j in plan['jobs']}
    if names and set(names) - known:
        raise ValueError(f'Unknown jobs: {sorted(set(names) - known)}; available: {sorted(known)}')
    selected = []
    for phase in ('train', 'eval', 'report'):
        for j in plan['jobs']:
            if j['stage'] != phase or (stage != 'all' and phase != stage):
                continue
            if names and j['name'] not in names:
                continue
            if j.get('extension') and not (extensions or names):
                continue
            args = [str(a).format(seed=seed) for a in j['args']]
            for flag, value in j.get('seed_overrides', {}).get(str(seed), {}).items():
                args[args.index(flag)+1] = str(value)
            selected.append((j, [sys.executable, str(ROOT / 'scripts' / j['script']), *args]))
    if not selected:
        raise ValueError('No jobs match this stage and selection')
    return selected

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rq', required=True, choices=['rq1', 'rq2', 'rq3', 'rq4'])
    p.add_argument('--seed', type=int, choices=[17, 29], default=17)
    p.add_argument('--stage', choices=['train', 'eval', 'report', 'all'], default='all')
    p.add_argument('--job', nargs='+', help='Select named jobs; training dependencies must already exist for eval/report')
    p.add_argument('--extensions', action='store_true', help='Include the RQ2 16-concept, mixed-domain, and SDXL studies')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    try:
        selected = commands(a.rq, a.seed, a.stage, a.job, a.extensions)
    except ValueError as e:
        p.error(str(e))
    for job, cmd in selected:
        print(f"[{a.rq}/{job['stage']}/{job['name']}] {shlex.join(cmd)}", flush=True)
        if not a.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
