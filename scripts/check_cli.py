#!/usr/bin/env python3
"""Check every CLI and all preset argument lists without loading checkpoints."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    count = 0
    for script in sorted((ROOT/'scripts').glob('*.py')):
        if script.name == Path(__file__).name:
            continue
        subprocess.run([sys.executable,str(script),'--help'],cwd=ROOT,env=env,
                       check=True,stdout=subprocess.DEVNULL)
        count += 1
    bootstrap = """
import argparse, runpy, sys
class Parsed(BaseException): pass
original = argparse.ArgumentParser.parse_args
def parse(self, args=None, namespace=None):
    original(self, args, namespace)
    raise Parsed()
argparse.ArgumentParser.parse_args = parse
script = sys.argv[1]
sys.argv = sys.argv[1:]
try:
    runpy.run_path(script, run_name='__main__')
except Parsed:
    pass
else:
    raise RuntimeError('CLI never reached parse_args')
"""
    for path in sorted((ROOT/'experiments').glob('rq*.json')):
        for job in json.loads(path.read_text())['jobs']:
            # Validate the real parser, then stop before executing the experiment.
            args = [str(x).format(seed=17) for x in job['args']]
            subprocess.run([sys.executable,'-c',bootstrap,str(ROOT/'scripts'/job['script']),*args],
                           cwd=ROOT,env=env,check=True,stdout=subprocess.DEVNULL)
            count += 1
    print(f'{count} CLI checks passed')

if __name__ == '__main__':
    main()
