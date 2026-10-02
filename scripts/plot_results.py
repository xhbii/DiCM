#!/usr/bin/env python3
"""Plot complete subset summaries on common linear axes."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary', help='summary.json produced by summarize.py')
    p.add_argument('--out', default='outputs/comparison.svg')
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports = json.loads(Path(a.summary).read_text())
    fig, axes = plt.subplots(1,4,figsize=(13,3),layout='constrained')
    for ax,key,label in zip(axes, ['erasure','unselected_retention','dino','clip'],
                           ['Selected-concept erasure','Unselected concepts retained','Retain DINO','Retain CLIP']):
        for report in reports:
            rows = [row for row in report['by_k'] if row[key] is not None]
            ax.plot([row['k'] for row in rows], [row[key] for row in rows], marker='o',
                    linewidth=1.5, markersize=4, label=report['run'])
        ax.set(xlabel='Installed modules (k)',ylabel=label,xticks=[1,2,4,6,8])
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=0.2)
        if key in ('erasure','unselected_retention','dino'):
            ax.set_ylim(0,1.03)
    axes[0].legend(fontsize=7)
    out = Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(out)

if __name__ == '__main__':
    main()
