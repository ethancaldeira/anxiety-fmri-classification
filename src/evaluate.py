"""
evaluate.py — post-hoc analysis of a finished training run.

Consumes the canonical bundle written by train.py:

    results/ch{N}/<EXPERIMENT_NAME>/
        ├── predictions_subject.csv   ← source of truth
        └── metrics_subject.json

Subcommands:

  run <dir>        per-run plot set (CM, ROC, PR, prob histogram,
                   calibration, borderline). Auto-invoked by train.py.

  compare <glob>   cross-run ROC overlay (one line per run), useful when
                   running batches across atlases or datasets.

  table <glob>     cross-run comparison TABLE (markdown + optional CSV),
                   one row per run with key hyperparams + headline metrics.

Examples:
    python src/evaluate.py run results/ch5/gsp_classification_voxel_cc200_roi_atlas
    python src/evaluate.py compare 'results/ch5/*classification*cc200_roi_atlas'
    python src/evaluate.py table 'results/ch5/*' --out results/ch5/_compare/table.csv
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc as sk_auc, confusion_matrix, precision_recall_curve,
    roc_curve,
)

import results_io


CLASS_NAMES = ['Negative', 'Positive']


# ---------------------------------------------------------------------------
# per-run plots (subject-level)
# ---------------------------------------------------------------------------

def _plot_confusion(y_true, y_pred, out_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def _plot_roc(y_true, y_prob, out_path, title):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = sk_auc(fpr, tpr)
    plt.figure(figsize=(5.5, 4.5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def _plot_pr(y_true, y_prob, out_path, title):
    if len(np.unique(y_true)) < 2:
        return
    from sklearn.metrics import average_precision_score
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(5.5, 4.5))
    plt.plot(rec, prec, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def _plot_prob_histogram(y_true, y_prob, out_path, title):
    """Histogram of predicted probabilities, split by true class."""
    plt.figure(figsize=(7, 4.5))
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.55, label='True Negative',
             color='tab:blue', edgecolor='white')
    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.55, label='True Positive',
             color='tab:red', edgecolor='white')
    plt.axvline(0.5, color='k', linestyle='--', alpha=0.5, label='threshold 0.5')
    plt.xlabel('Predicted P(positive)'); plt.ylabel('Count')
    plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def _plot_calibration(y_true, y_prob, out_path, title, n_bins=10):
    """Reliability diagram — observed positive rate per predicted-prob bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    centres, fracs, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi if hi < 1 else y_prob <= hi)
        if m.sum() == 0:
            continue
        centres.append((lo + hi) / 2)
        fracs.append(float(y_true[m].mean()))
        counts.append(int(m.sum()))
    if not centres:
        return
    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='perfect calibration')
    ax1.plot(centres, fracs, 'o-', color='tab:purple', label='observed')
    ax1.set_xlabel('Predicted P(positive)'); ax1.set_ylabel('Observed fraction positive')
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.grid(alpha=0.3); ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.bar(centres, counts, width=0.08, alpha=0.2, color='tab:purple')
    ax2.set_ylabel('Subjects per bin', color='tab:purple')
    plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def _plot_borderline(subj_df, out_path, title, band=0.05):
    """Subjects sorted by predicted prob, coloured by true class. A
    horizontal band of width ±`band` around 0.5 highlights uncertain
    predictions."""
    df = subj_df.sort_values('prob').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.18 * len(df))))
    for cls, colour, lab in [(0, 'tab:blue', 'Negative'), (1, 'tab:red', 'Positive')]:
        mask = df['true'] == cls
        ax.scatter(df.loc[mask, 'prob'], df.index[mask],
                   color=colour, s=40, edgecolor='k', linewidth=0.4,
                   label=f'True {lab} (n={int(mask.sum())})')
    ax.axvline(0.5, color='k', linestyle='--', alpha=0.5)
    ax.axvspan(0.5 - band, 0.5 + band, color='yellow', alpha=0.2,
               label=f'borderline band (±{band})')
    n_borderline = int(((df['prob'] >= 0.5 - band) & (df['prob'] <= 0.5 + band)).sum())
    ax.set_xlabel("Predicted P(positive)"); ax.set_ylabel('Test subject (sorted)')
    ax.set_xlim(0, 1)
    ax.set_title(f'{title}  —  {n_borderline}/{len(df)} subjects in borderline band')
    ax.grid(axis='x', alpha=0.3); ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _plot_training_history(run_dir):
    """Loss + AUC over epochs from training_history.json."""
    path = os.path.join(run_dir, 'training_history.json')
    if not os.path.isfile(path):
        return
    with open(path) as f:
        payload = json.load(f)
    h = payload.get('history', {})
    if not h:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in zip(axes, ['loss', 'auc'], ['Loss', 'AUC']):
        if key not in h:
            continue
        epochs = range(1, len(h[key]) + 1)
        ax.plot(epochs, h[key], 'bo-', label=f'train {title}')
        if f'val_{key}' in h:
            ax.plot(epochs, h[f'val_{key}'], 'ro-', label=f'val {title}')
        ax.set_xlabel('Epoch'); ax.set_ylabel(title); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, 'plots', 'training_curves.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# evaluate_run
# ---------------------------------------------------------------------------

def evaluate_run(run_dir):
    """Produce the full plot set + (re-)write metrics for one run dir."""
    csv_path = os.path.join(run_dir, 'predictions_subject.csv')
    if not os.path.isfile(csv_path):
        print(f"  [skip] {run_dir} — no predictions_subject.csv")
        return None

    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    metrics = results_io.write_metrics_classification(run_dir, df)
    name = os.path.basename(run_dir.rstrip('/'))

    y_true = df['true'].values
    y_prob = df['prob'].values
    y_pred = df['pred'].values

    _plot_training_history(run_dir)
    _plot_confusion(y_true, y_pred, os.path.join(plots_dir, 'confusion_matrix.png'),
                    title=f'{name}  (n={len(df)})')
    _plot_roc(y_true, y_prob, os.path.join(plots_dir, 'roc_curve.png'),
              title=f'{name} — ROC')
    _plot_pr(y_true, y_prob, os.path.join(plots_dir, 'pr_curve.png'),
             title=f'{name} — PR')
    _plot_prob_histogram(y_true, y_prob, os.path.join(plots_dir, 'prob_histogram.png'),
                         title=f'{name} — predicted probability distribution')
    _plot_calibration(y_true, y_prob, os.path.join(plots_dir, 'calibration.png'),
                      title=f'{name} — calibration')
    _plot_borderline(df, os.path.join(plots_dir, 'borderline_analysis.png'),
                     title=f'{name} — borderline analysis')

    print(f"  {name:<55s}  acc={metrics['accuracy']:.3f}  "
          f"auc={metrics['roc_auc']:.3f}  "
          f"prec={metrics['precision']:.3f}  rec={metrics['recall']:.3f}")
    return metrics


# ---------------------------------------------------------------------------
# cross-run ROC overlay
# ---------------------------------------------------------------------------

def _plot_compare_roc(run_dirs, out_path,
                      title='ROC overlay across runs'):
    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = 0
    for rd in sorted(run_dirs):
        csv_path = os.path.join(rd, 'predictions_subject.csv')
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if df['true'].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(df['true'], df['prob'])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, alpha=0.85,
                label=f"{os.path.basename(rd)}  AUC={roc_auc:.3f}")
        plotted += 1
    if plotted == 0:
        print(f"  [skip] compare — no runs with predictions_subject.csv")
        plt.close(fig); return
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.set_title(title); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}  ({plotted} runs overlaid)")


# ---------------------------------------------------------------------------
# cross-run comparison table
# ---------------------------------------------------------------------------

TABLE_DEFAULT_HYPERPARAMS = [
    'DATASET', 'TASK', 'CONNECTIVITY_TYPE', 'ATLAS', 'STAI_CUTOFF',
]
TABLE_METRICS = ['accuracy', 'precision', 'recall', 'roc_auc']


def _load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _fmt_metric(level_metrics, key):
    if level_metrics is None:
        return '—'
    v = level_metrics.get(key)
    if v is None:
        return '—'
    return f"{float(v):.3f}"


def _row_for_run(run_dir, hyperparams):
    cfg = _load_json(os.path.join(run_dir, 'config.json')) or {}
    m = _load_json(os.path.join(run_dir, 'metrics_subject.json'))
    row = {'run': os.path.basename(os.path.normpath(run_dir))}
    for hp in hyperparams:
        row[hp] = cfg.get(hp, '—')
    for k in TABLE_METRICS:
        row[k] = _fmt_metric(m, k)
    return row


def _sort_key(value):
    if value == '—':
        return float('-inf')
    return float(value)


def build_comparison_table(pattern, hyperparams=None, sort_by='roc_auc', out_csv=None):
    hyperparams = hyperparams or TABLE_DEFAULT_HYPERPARAMS
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    rows = []
    for d in dirs:
        if not os.path.isfile(os.path.join(d, 'config.json')):
            continue
        rows.append(_row_for_run(d, hyperparams))
    if not rows:
        raise SystemExit(f"No valid runs (config.json) under {pattern!r}")
    df = pd.DataFrame(rows)
    df = df.iloc[df[sort_by].map(_sort_key).argsort()[::-1]].reset_index(drop=True)
    print(df.to_markdown(index=False))
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nWrote {out_csv}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest='cmd', required=True)

    ar = sub.add_parser('run', help='per-run plot set')
    ar.add_argument('run_dir', help='Path to results/ch{N}/<EXPERIMENT_NAME>')

    ac = sub.add_parser('compare', help='cross-run ROC overlay')
    ac.add_argument('pattern', help='Glob of run dirs, e.g. "results/ch5/*cc200*"')

    at = sub.add_parser('table', help='cross-run comparison table')
    at.add_argument('pattern', help='Glob of run dirs, e.g. "results/ch5/*"')
    at.add_argument('--hyperparams', nargs='+', default=TABLE_DEFAULT_HYPERPARAMS)
    at.add_argument('--sort', default='roc_auc', choices=TABLE_METRICS)
    at.add_argument('--out', help='Optional CSV output path')

    args = ap.parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.cmd == 'run':
        evaluate_run(args.run_dir)
    elif args.cmd == 'compare':
        out = os.path.join(repo_root, 'results', '_compare', 'roc_overlay.png')
        run_dirs = sorted(d for d in glob.glob(args.pattern) if os.path.isdir(d))
        _plot_compare_roc(run_dirs, out)
    elif args.cmd == 'table':
        build_comparison_table(args.pattern, hyperparams=args.hyperparams,
                               sort_by=args.sort, out_csv=args.out)


if __name__ == '__main__':
    main()
