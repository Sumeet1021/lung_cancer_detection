"""
train_efficientnet.py — EfficientNetB0 | Two-Phase Fine-Tuning
==============================================================
Compatible with TF 2.16.x through 2.20.x on Python 3.12
Uses tf_keras package — the permanent fix for standalone keras conflicts.

Install once before running:
    pip install tf_keras
"""

import os, json, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info/warning spam

import numpy as np
import tensorflow as tf

# ── tf_keras: stable Keras 2 API, works with any TF 2.16+ ────────────────────
# Install: pip install tf_keras
import tf_keras
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.applications import EfficientNetB0
from tf_keras.applications.efficientnet import preprocess_input
from tf_keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tf_keras.models import Model
from tf_keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

print(f"TensorFlow : {tf.__version__}")
print(f"tf_keras   : {tf_keras.__version__}")

# ── REPRODUCIBILITY ───────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_DIR  = "dataset/images"
MODEL_PATH = "models/efficientnet_model.h5"
BEST_PATH  = "models/efficientnet_best.h5"
ACC_PATH   = "models/accuracy.json"

IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
EPOCHS     = 10

os.makedirs("models", exist_ok=True)

# ── DATA PIPELINE ─────────────────────────────────────────────────────────────
# Mild augmentation — extreme values hurt the small benign class (120 images)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.85, 1.15],
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb",
    subset="training",
    shuffle=True,
    seed=SEED
)

val_data = val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb",
    subset="validation",
    shuffle=False,
    seed=SEED
)

num_classes = train_data.num_classes
print(f"\nClasses      : {train_data.class_indices}")
print(f"Train samples: {train_data.samples}")
print(f"Val samples  : {val_data.samples}")

# ── CLASS WEIGHTS ─────────────────────────────────────────────────────────────
# balanced only — no extra multipliers (they cause gradient instability)
labels            = train_data.classes
cw                = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(cw))
print(f"Class weights: {class_weight_dict}\n")

# ── MODEL ─────────────────────────────────────────────────────────────────────
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False    # Phase 1: freeze entire base

x      = base_model.output
x      = GlobalAveragePooling2D()(x)
x      = BatchNormalization()(x)
x      = Dense(256, activation="relu")(x)
x      = Dropout(0.5)(x)
x      = Dense(128, activation="relu")(x)
x      = Dropout(0.4)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ── PHASE 1: TRAIN HEAD ONLY ──────────────────────────────────────────────────
# Base is frozen → high LR is safe here, head learns from scratch
print("=" * 55)
print("PHASE 1 — Head only, base frozen (3 epochs, LR=1e-3)")
print("=" * 55)

model.compile(
    optimizer=tf_keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf_keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=3,
    class_weight=class_weight_dict,
    callbacks=[
        ModelCheckpoint(BEST_PATH, monitor="val_accuracy",
                        save_best_only=True, mode="max", verbose=1)
    ],
    verbose=1
)

# ── PHASE 2: FINE-TUNE TOP-50 BASE LAYERS ────────────────────────────────────
# Unfreeze top 50 layers, keep BN frozen, use low LR to avoid destroying weights
print("\n" + "=" * 55)
print("PHASE 2 — Top-50 unfrozen (7 epochs, LR=5e-5)")
print("=" * 55)

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False
# Freeze all BatchNorm in base — updating BN stats with small data is harmful
for layer in base_model.layers:
    if isinstance(layer, tf_keras.layers.BatchNormalization):
        layer.trainable = False

trainable = sum(1 for l in model.layers if l.trainable)
print(f"Trainable layers: {trainable}")

model.compile(
    optimizer=tf_keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf_keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS - 3,          # 7 remaining epochs
    class_weight=class_weight_dict,
    callbacks=[
        ModelCheckpoint(BEST_PATH, monitor="val_accuracy",
                        save_best_only=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                         patience=2, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=4,
                      restore_best_weights=True, verbose=1)
    ],
    verbose=1
)

# ── RESULTS ───────────────────────────────────────────────────────────────────
all_train = history1.history["accuracy"]     + history2.history["accuracy"]
all_val   = history1.history["val_accuracy"] + history2.history["val_accuracy"]
best_train = round(float(max(all_train)) * 100, 2)
best_val   = round(float(max(all_val))   * 100, 2)

print(f"\n{'='*55}")
print(f"Best Train Accuracy : {best_train:.2f}%")
print(f"Best Val   Accuracy : {best_val:.2f}%")
print(f"{'='*55}")

# ── SAVE MODEL ────────────────────────────────────────────────────────────────
try:
    best_model = tf_keras.models.load_model(BEST_PATH, compile=False)
    best_model.save(MODEL_PATH)
    print(f"\nBest checkpoint → {MODEL_PATH}")
except Exception as e:
    model.save(MODEL_PATH)
    print(f"Saved current model (best load failed: {e})")

# ── SAVE ACCURACY JSON (read by app.py dashboard) ─────────────────────────────
with open(ACC_PATH, "w") as f:
    json.dump({"train_accuracy": best_train, "val_accuracy": best_val}, f, indent=2)

print(f"Accuracy JSON: {ACC_PATH}")
print("\n✅ Training complete!")