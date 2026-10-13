"""
model.py — models and callbacks for the rs-fMRI classification pipeline.

Two architectures, picked by Config.CONNECTIVITY_TYPE:

  voxel    -> Classifier3DCNN on (31, 37, 31, N_ROIs) supervoxel fingerprints
  roi_roi  -> ClassifierMLP    on the flattened upper triangle of an
                                 (N_ROIs, N_ROIs) Fisher-z connectivity matrix

Both classifiers follow the Khosla et al. (Ensemble3DCNN_connectomes)
recipe.

Only TASK='classification' is implemented. Another task (e.g. regression
on continuous STAI-T) would add a model head here, a predictions/metrics
writer in results_io.py, and a target column in the dataset adapter.
"""

import os

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Conv3D, Dense, Dropout,
    Flatten, Input, MaxPooling3D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD


# ---------------------------------------------------------------------------
# model definitions (Khosla et al. recipe)
# ---------------------------------------------------------------------------

def Classifier3DCNN(input_shape):
    """3D CNN binary classifier for voxel-wise connectivity fingerprints.

    Conv3D(128) -> ELU -> Conv3D(32) -> ELU -> MaxPool3D
    -> Flatten -> Dense(32, ELU, L2) -> Dense(1, sigmoid)
    """
    if len(input_shape) != 4:
        raise ValueError(f"Voxel input must be 4-D (x, y, z, n_rois); got {input_shape}")
    model = Sequential(name='Classifier3DCNN')
    model.add(Conv3D(128, kernel_size=(3, 3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('elu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('elu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(32, activation='elu', kernel_regularizer=regularizers.l2(0.005)))
    model.add(Dense(1, activation='sigmoid'))
    return model


def ClassifierMLP(input_shape):
    """MLP binary classifier for 1-D ROI-ROI flattened connectivity.

    Dense(256, ELU, L2) -> BN -> Dropout(0.3)
    -> Dense(64, ELU, L2) -> BN -> Dropout(0.3)
    -> Dense(32, ELU)     -> Dense(1, sigmoid)
    """
    model = Sequential(name='ClassifierMLP')
    model.add(Input(shape=input_shape))
    model.add(Dense(256, activation='elu', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='elu', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


# ---------------------------------------------------------------------------
# build + compile
# ---------------------------------------------------------------------------

def build_model(input_shape, config):
    """Pick + compile a model based on Config.CONNECTIVITY_TYPE / TASK."""
    if config.CONNECTIVITY_TYPE == 'voxel':
        model = Classifier3DCNN(input_shape)
    elif config.CONNECTIVITY_TYPE == 'roi_roi':
        model = ClassifierMLP(input_shape)
    else:
        raise ValueError(f"Unknown CONNECTIVITY_TYPE: {config.CONNECTIVITY_TYPE}")

    if config.TASK != 'classification':
        raise NotImplementedError(
            f"TASK={config.TASK!r}: only 'classification' is implemented."
        )

    # Khosla recipe: SGD + momentum 0.9 + per-step decay 1e-3 by default,
    # i.e. lr_t = lr / (1 + decay * t), expressed as an InverseTimeDecay
    # schedule.
    decay = getattr(config, 'SGD_DECAY', 0.0) or 0.0
    if decay > 0:
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=config.LEARNING_RATE,
            decay_steps=1,
            decay_rate=decay,
        )
        opt = SGD(learning_rate=lr_schedule, momentum=0.9)
    else:
        opt = SGD(learning_rate=config.LEARNING_RATE, momentum=0.9)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )
    trainable = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
    print(f"  Model: {model.name}   total params: {model.count_params():,}  "
          f"trainable: {trainable:,}")
    return model


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------

def get_callbacks(config, run_dir):
    """EarlyStopping (val_loss, restore best) + ModelCheckpoint to best.keras."""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            os.path.join(run_dir, 'best.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
    ]
