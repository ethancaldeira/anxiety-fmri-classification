"""
train.py — single entry point for both GSP and ABIDE classification.

Runs:
    python src/train.py

Reads Config from src/config.py (env vars override most knobs). Dispatches
to the right dataset adapter via Config.DATASET, loads features, splits
by subject, builds a 3D CNN or MLP per Config.CONNECTIVITY_TYPE, trains
for EPOCHS epochs, restores the lowest-val_loss weights, evaluates on the
held-out test set, and writes the canonical
results/ch{N}/<EXPERIMENT_NAME>/ bundle.

evaluate.evaluate_run is called automatically at the end so every run
gets the full plot set without a separate command (mirrors Ch4).
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import gc
import math
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

# Make src/ importable when run as `python src/train.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from datasets import get_adapter
import data_loader
import model as model_module
import results_io
from evaluate import evaluate_run


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _ArraySequence(tf.keras.utils.Sequence):
    """Batch streamer for numpy arrays — keeps X in CPU RAM, sends one
    batch at a time to the GPU.

    Critical for voxel-wise GSP: the full training-set array is ~16 GB
    (595 subjects × 31×37×31×200 × 4 bytes), which won't fit on the
    Tesla T4's 16 GB VRAM. tf.data.Dataset.from_tensor_slices ALSO
    fails here because it materialises X as a single tf.constant on
    GPU. A Sequence subclass is the only path that genuinely streams
    (Keras pulls one batch via __getitem__, that batch is the only
    thing copied to GPU memory).
    """

    def __init__(self, X, y, batch_size, shuffle=False, seed=42):
        super().__init__()
        self.X = X
        self.y = np.asarray(y, dtype=np.float32)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self._rng = np.random.default_rng(seed)
        self._idx = np.arange(len(X))
        if self.shuffle:
            self._rng.shuffle(self._idx)

    def __len__(self):
        return int(math.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, i):
        sl = self._idx[i * self.batch_size : (i + 1) * self.batch_size]
        return self.X[sl], self.y[sl]

    def on_epoch_end(self):
        if self.shuffle:
            self._rng.shuffle(self._idx)


def _train_and_evaluate(data, run_dir):
    """Train + evaluate one model on one (train, val, test) bundle.
    Writes the canonical results bundle to run_dir. Returns metrics dict.
    """
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
    results_io.write_config_snapshot(Config, run_dir)

    X_tr, y_tr = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_te, y_te, sid_te = data['X_test'], data['y_test'], data['sid_test']

    print(f"\n  Building model for input shape {X_tr.shape[1:]} ...")
    model = model_module.build_model(X_tr.shape[1:], Config)

    callbacks = list(model_module.get_callbacks(Config, run_dir))

    print(f"\n--- Training —  train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}")

    # Stream batches from CPU RAM so the 16 GB GSP train tensor never has to
    # fit on the GPU (T4 has only 16 GB VRAM).
    train_ds = _ArraySequence(X_tr,  y_tr,  Config.BATCH_SIZE, shuffle=True)
    val_ds   = _ArraySequence(X_val, y_val, Config.BATCH_SIZE, shuffle=False)
    test_ds  = _ArraySequence(X_te,  y_te,  Config.BATCH_SIZE, shuffle=False)

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    train_secs = time.time() - t0
    results_io.write_training_history(run_dir, history, train_secs)

    # ---- Test predictions: source-of-truth CSV + metrics + plots ----
    probs = model.predict(test_ds, verbose=0).ravel()
    pred_df = results_io.write_predictions_classification(run_dir, sid_te, y_te, probs)
    metrics = results_io.write_metrics_classification(run_dir, pred_df)

    print(f"\n  TEST (subject-level):  acc={metrics['accuracy']:.3f}  "
          f"auc={metrics['roc_auc']:.3f}  "
          f"prec={metrics['precision']:.3f}  rec={metrics['recall']:.3f}")

    # Full plot set via evaluate_run (auto-invoked, no separate command).
    evaluate_run(run_dir)

    tf.keras.backend.clear_session()
    del model
    gc.collect()
    return metrics


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print('=' * 60)
    Config.print_config()
    Config.make_run_dirs()

    adapter = get_adapter(Config.DATASET, Config)
    bundle  = data_loader.load_and_prepare(Config, adapter)

    _train_and_evaluate(bundle, Config.RUN_DIR)

    # Cache cleanup — multi-atlas batches set KEEP_CACHE=0 so each run
    # removes its per-subject .npy features (tens of GB for GSP voxel
    # features). Default is to keep them so a single cell can be repeated
    # without recomputing or downloading features.
    if not getattr(Config, 'KEEP_CACHE', True):
        cache_dir = adapter.local_cache_dir(Config.ATLAS, Config.CONNECTIVITY_TYPE)
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"\n[cleanup] removed feature cache: {cache_dir}")

    print(f"\nResults saved to: {Config.RUN_DIR}")


if __name__ == '__main__':
    main()
