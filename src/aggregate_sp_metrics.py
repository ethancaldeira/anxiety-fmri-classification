"""
aggregate_sp_metrics.py — per-scale distribution stats for the stochastic
parcellation (SP) set: one observation per stochastic sample, summarised
per scale. The soft-vote ensemble of the samples at each scale is written
separately by aggregate_ensemble.py (SP_classification/<dataset>/_ensemble/).

For each (dataset, connectivity), reads every
    results/ch{N}/SP_classification/{dataset}/{dataset}_classification_{conn}_SP_v2_{scale}_{sample}/
metrics_subject.json and emits:
    _summary/
        {conn}_per_sample.csv       one row per (scale, sample): acc/AUC/prec/rec
        {conn}_per_scale.csv        one row per scale: n, mean ± std ± 95%CI per metric
        {conn}_per_scale.md         markdown table for thesis paste-in
        plots/
            {conn}_auc_by_scale.png boxplot, scale on x

Usage:
    python src/aggregate_sp_metrics.py --chapter 6 --dataset gsp --connectivity roi_roi
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

METRICS = ['accuracy', 'roc_auc', 'precision', 'recall']

RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def _collect(chapter: str, dataset: str, conn: str):
    base = os.path.join(RESULTS_ROOT, f'ch{chapter}', 'SP_classification', dataset)
    if not os.path.isdir(base):
        sys.exit(f"ERROR: not found: {base}")

    pattern = re.compile(
        rf"^{re.escape(dataset)}_classification_{re.escape(conn)}_"
        rf"SP_v2_(\d{{3}})_(\d{{2}})$"
    )
    rows = []
    for d in sorted(os.listdir(base)):
        m = pattern.match(d)
        if not m: continue
        scale, sample = int(m.group(1)), int(m.group(2))
        mp = os.path.join(base, d, 'metrics_subject.json')
        if not os.path.isfile(mp): continue
        rec = json.load(open(mp))
        rows.append({
            'scale':    scale,
            'sample':   sample,
            'accuracy': rec.get('accuracy'),
            'roc_auc':  rec.get('roc_auc'),
            'precision':rec.get('precision'),
            'recall':   rec.get('recall'),
            'n':        rec.get('n'),
        })
    return pd.DataFrame(rows)


def _per_scale_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Group by scale, compute n / mean / std / 95% CI per metric."""
    out = []
    for scale, g in df.groupby('scale'):
        row = {'scale': scale, 'n_samples': len(g)}
        for m in METRICS:
            vals = g[m].dropna().values
            if len(vals) == 0:
                continue
            mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1))
            # 95% CI via normal approx (n>=10 ok); for n<10 use t-distribution
            # but at our sample sizes the difference is tiny.
            ci = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            row[f'{m}_mean'] = mean
            row[f'{m}_std']  = std
            row[f'{m}_ci95'] = ci
        out.append(row)
    return pd.DataFrame(out).sort_values('scale').reset_index(drop=True)


def _markdown_table(summary: pd.DataFrame) -> str:
    lines = [
        '| Scale | n | accuracy | AUC | precision | recall |',
        '|---|---|---|---|---|---|',
    ]
    for _, r in summary.iterrows():
        fmt = lambda m: f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_std']:.3f}"
        lines.append(
            f"| {int(r.scale)} | {int(r.n_samples)} | "
            f"{fmt('accuracy')} | {fmt('roc_auc')} | "
            f"{fmt('precision')} | {fmt('recall')} |"
        )
    return '\n'.join(lines) + '\n'


def _boxplot(df: pd.DataFrame, metric: str, out_path: str, dataset: str, conn: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scales = sorted(df['scale'].unique())
    data = [df[df.scale == s][metric].dropna().values for s in scales]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, tick_labels=[f'{s}' for s in scales], showmeans=True)
    ax.set_xlabel('Number of ROIs (scale)')
    ax.set_ylabel(metric.upper() if metric == 'roc_auc' else metric.capitalize())
    ax.set_title(f'SP {dataset.upper()} ({conn}) — {metric} across stochastic samples')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chapter',      required=True)
    ap.add_argument('--dataset',      required=True, choices=['gsp', 'abide'])
    ap.add_argument('--connectivity', required=True, choices=['roi_roi', 'voxel'])
    args = ap.parse_args()

    df = _collect(args.chapter, args.dataset, args.connectivity)
    if df.empty:
        sys.exit(f"ERROR: no SP cells found for ch{args.chapter}/{args.dataset}/{args.connectivity}")

    summary = _per_scale_summary(df)
    md = _markdown_table(summary)

    out_dir = os.path.join(RESULTS_ROOT, f'ch{args.chapter}', 'SP_classification',
                           args.dataset, '_summary')
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)

    df.sort_values(['scale', 'sample']).to_csv(
        os.path.join(out_dir, f'{args.connectivity}_per_sample.csv'), index=False)
    summary.to_csv(
        os.path.join(out_dir, f'{args.connectivity}_per_scale.csv'), index=False)
    with open(os.path.join(out_dir, f'{args.connectivity}_per_scale.md'), 'w') as f:
        f.write(md)

    _boxplot(df, 'roc_auc',
             os.path.join(out_dir, 'plots', f'{args.connectivity}_auc_by_scale.png'),
             args.dataset, args.connectivity)

    print(f"\n{'=' * 70}", flush=True)
    print(f"SP METRIC SUMMARY — ch{args.chapter} / {args.dataset} / {args.connectivity}",
          flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  n_cells: {len(df)}  (scales: {sorted(df['scale'].unique())})", flush=True)
    print(f"  per-scale mean ± std (AUC):", flush=True)
    for _, r in summary.iterrows():
        print(f"    scale={int(r.scale)}  n={int(r.n_samples)}  "
              f"AUC={r['roc_auc_mean']:.3f} ± {r['roc_auc_std']:.3f}", flush=True)
    print(f"  written to: {out_dir}", flush=True)


if __name__ == '__main__':
    main()
