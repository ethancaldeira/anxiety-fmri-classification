"""
results_io.py — write the canonical per-run output bundle.

Mirrors the Ch4 (anxiety-smri-classification) layout, adapted for
subject-level predictions (the natural unit in Ch5):

    results/ch{N}/<EXPERIMENT_NAME>/
        ├── config.json
        ├── best.keras
        ├── predictions_subject.csv   ← source of truth (one row per subject)
        ├── metrics_subject.json
        ├── training_history.json
        └── plots/

predictions_subject.csv columns:
    subject_id, true, prob, pred

Metric set (matches Ch4 + thesis): accuracy, precision, recall, roc_auc
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
)


# ---------- config snapshot ------------------------------------------------

def write_config_snapshot(config, run_dir):
    """Dump JSON-serialisable Config attributes to config.json."""
    snapshot = {}
    for k in dir(config):
        if k.startswith('_') or callable(getattr(config, k)):
            continue
        v = getattr(config, k)
        try:
            json.dumps(v)
            snapshot[k] = v
        except TypeError:
            snapshot[k] = repr(v)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)


# ---------- predictions CSV (source of truth) ------------------------------

def write_predictions_classification(run_dir, subject_ids, y_true, y_prob,
                                     threshold=0.5):
    """One row per subject for classification. Returns the DataFrame."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    df = pd.DataFrame({
        'subject_id': list(subject_ids),
        'true':       y_true,
        'prob':       y_prob,
        'pred':       y_pred,
    })
    df.to_csv(os.path.join(run_dir, 'predictions_subject.csv'), index=False)
    return df


# ---------- metrics --------------------------------------------------------

def _safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_prob))


def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    """Classification metric set."""
    y_pred = (np.asarray(y_prob).ravel() >= threshold).astype(int)
    return {
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'roc_auc':   _safe_auc(y_true, y_prob),
        'n':         int(len(y_true)),
    }


def write_metrics_classification(run_dir, predictions_df):
    """Compute + write metrics_subject.json from predictions_subject.csv."""
    m = compute_classification_metrics(predictions_df['true'].values,
                                       predictions_df['prob'].values)
    with open(os.path.join(run_dir, 'metrics_subject.json'), 'w') as f:
        json.dump(m, f, indent=2)
    return m


# ---------- training history ----------------------------------------------

def write_training_history(run_dir, history, training_time_seconds=None):
    """Serialise a Keras History.history dict to JSON."""
    payload = {
        'training_time_seconds': training_time_seconds,
        'n_epochs': len(next(iter(history.history.values()))) if history.history else 0,
        'history': {k: [float(v) for v in vs] for k, vs in history.history.items()},
    }
    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump(payload, f, indent=2)
