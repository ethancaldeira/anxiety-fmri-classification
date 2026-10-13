"""
aggregate_ensemble.py — combine N classification cells into one soft-vote
ensemble by averaging per-subject probabilities (threshold 0.5).

Used in Ch6 for
  - the group parcellations: one model per train-derived tertile mask
    (group_gm_low/mid/high), all applied to every subject;
  - the stochastic parcellations: the SP_SAMPLES_PER_K samples at each
    scale (results/ch6/SP_classification/<dataset>/_ensemble/).
run_experiments.py calls it at the end of run_group_ensemble and
run_sp_atlases.

Inputs:
    Component result dirs (relative to results/ch{N}/), each containing
    predictions_subject.csv with columns: subject_id, true, prob, pred.
    All components must share the same test subjects (they use the same
    split).

Outputs:
    results/ch{N}/<output_name>/
        ├── predictions_subject.csv   ← subject_id, true, prob, pred
        │                               (prob = mean of component probs)
        ├── metrics_subject.json      ← accuracy, precision, recall, roc_auc, n
        └── components.json           ← list of component run dirs +
                                        their individual test AUCs

Usage:
    python src/aggregate_ensemble.py \\
        --chapter 6 \\
        --output-name gsp_classification_roi_roi_group_gm_ensemble \\
        --components gsp_classification_roi_roi_group_gm_low \\
                     gsp_classification_roi_roi_group_gm_mid \\
                     gsp_classification_roi_roi_group_gm_high
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_io import compute_classification_metrics  # noqa: E402

RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def ensemble(comp_dirs: list[str], components: list[str]):
    """Average per-subject probabilities across component run dirs.

    Returns (predictions DataFrame, metrics dict, per-component test AUCs).
    """
    for d in comp_dirs:
        p = os.path.join(d, 'predictions_subject.csv')
        if not os.path.isfile(p):
            sys.exit(f"ERROR: missing component predictions: {p}")

    # Read each component's predictions, keep prob renamed to prob_<component>.
    dfs = []
    comp_aucs = {}
    for c, d in zip(components, comp_dirs):
        df = pd.read_csv(os.path.join(d, 'predictions_subject.csv'))
        comp_aucs[c] = float(roc_auc_score(df['true'], df['prob']))
        df = df[['subject_id', 'true', 'prob']].rename(columns={'prob': f'prob_{c}'})
        dfs.append(df)

    # Merging on subject_id AND true also checks label consistency.
    merged = dfs[0]
    for df in dfs[1:]:
        before = len(merged)
        merged = merged.merge(df, on=['subject_id', 'true'], how='inner')
        if len(merged) != before:
            print(f"  [warn] subject overlap shrank {before} -> {len(merged)} "
                  f"after merging with a component", flush=True)

    prob_cols = [c for c in merged.columns if c.startswith('prob_')]
    merged['prob'] = merged[prob_cols].mean(axis=1)
    merged['pred'] = (merged['prob'] >= 0.5).astype(int)

    metrics = compute_classification_metrics(merged['true'].values,
                                             merged['prob'].values)
    return merged[['subject_id', 'true', 'prob', 'pred']], metrics, comp_aucs


def aggregate(chapter: str, output_name: str, components: list[str]) -> None:
    base = os.path.join(RESULTS_ROOT, f'ch{chapter}')
    comp_dirs = [os.path.join(base, c) for c in components]
    out_pred, metrics, comp_aucs = ensemble(comp_dirs, components)

    out_dir = os.path.join(base, output_name)
    os.makedirs(out_dir, exist_ok=True)

    out_pred.to_csv(os.path.join(out_dir, 'predictions_subject.csv'), index=False)
    with open(os.path.join(out_dir, 'metrics_subject.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, 'components.json'), 'w') as f:
        json.dump({
            'components':         components,
            'component_test_aucs': comp_aucs,
            'aggregation':        'mean of per-subject prob across components',
            'ensemble_test_auc':  metrics['roc_auc'],
        }, f, indent=2)

    print(f"\n{'=' * 70}", flush=True)
    print(f"ENSEMBLE — {output_name}", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  n_subjects: {metrics['n']}", flush=True)
    print(f"  components ({len(components)}):", flush=True)
    for c, auc in comp_aucs.items():
        print(f"    {c:55s} test AUC={auc:.3f}", flush=True)
    print(f"  ENSEMBLE      acc={metrics['accuracy']:.3f}  "
          f"AUC={metrics['roc_auc']:.3f}  "
          f"prec={metrics['precision']:.3f}  "
          f"rec={metrics['recall']:.3f}", flush=True)
    print(f"  written to: {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chapter', required=True,
                    help='Chapter number (looks under results/ch{chapter}/).')
    ap.add_argument('--output-name', required=True,
                    help='Output dir name (created under results/ch{chapter}/).')
    ap.add_argument('--components', nargs='+', required=True,
                    help='Component run-dir names (must already exist).')
    args = ap.parse_args()
    aggregate(args.chapter, args.output_name, args.components)


if __name__ == '__main__':
    main()
