import os
import io
import time
import zipfile
import numpy as np
from PIL import Image
import tensorflow as tf

print("=" * 65)
print("   TRAINING GENUINE LUNGGUARD AI EFFICIENTNETB0 MODEL")
print("=" * 65)

ZIP_PATH = r"C:\Users\91770\Downloads\The IQ-OTHNCCD lung cancer dataset.zip"
TARGET_MODEL_PATH = r"C:\Users\91770\OneDrive\Desktop\Harsha_LUNG_AI\models\LungGuard_AI_Final.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Benign", "Malignant", "Normal"]  # Alphabetical index 0, 1, 2

# 1. Read Dataset from Zip
print("\n[*] Loading and splitting IQ-OTH-NCCD dataset...")
class_mapping = {
    "Bengin cases": 0,
    "Malignant cases": 1,
    "Normal cases": 2
}

all_images = {0: [], 1: [], 2: []}

with zipfile.ZipFile(ZIP_PATH) as outer:
    inner_bytes = outer.read("The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset.zip")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        files = inner.namelist()
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            for folder, cls_idx in class_mapping.items():
                if f"/{folder}/" in f:
                    all_images[cls_idx].append(inner.read(f))
                    break

# Deterministic Train/Val/Test Split matching Cell 15 (70% train, 15% val, 15% test)
train_data, train_labels = [], []
val_data, val_labels = [], []
test_data, test_labels = [], []

rng = np.random.RandomState(42)

for cls_idx in [0, 1, 2]:
    imgs = all_images[cls_idx]
    n = len(imgs)
    indices = rng.permutation(n)
    
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    print(f"  {CLASS_NAMES[cls_idx]}: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    train_data.extend([imgs[i] for i in train_idx])
    train_labels.extend([cls_idx] * len(train_idx))
    val_data.extend([imgs[i] for i in val_idx])
    val_labels.extend([cls_idx] * len(val_idx))
    test_data.extend([imgs[i] for i in test_idx])
    test_labels.extend([cls_idx] * len(test_idx))

print(f"\nTotal Splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

# Convert image bytes to preprocessed numpy tensors [0, 255]
def process_images(byte_list):
    arr = np.zeros((len(byte_list), 224, 224, 3), dtype=np.float32)
    for i, b in enumerate(byte_list):
        img = Image.open(io.BytesIO(b)).convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
        arr[i] = np.asarray(img, dtype=np.float32)
    return arr

print("\n[*] Processing images into (224, 224, 3) arrays...")
t0 = time.time()
X_train = process_images(train_data)
y_train = tf.keras.utils.to_categorical(np.array(train_labels), 3)

X_val = process_images(val_data)
y_val = tf.keras.utils.to_categorical(np.array(val_labels), 3)

X_test = process_images(test_data)
y_test = tf.keras.utils.to_categorical(np.array(test_labels), 3)
print(f"Processed all images in {time.time()-t0:.2f}s.")

# Compute Class Weights: N / (3 * N_c)
total_train = len(train_labels)
class_weight_dict = {
    0: total_train / (3.0 * train_labels.count(0)),
    1: total_train / (3.0 * train_labels.count(1)),
    2: total_train / (3.0 * train_labels.count(2))
}
print("Class weights:", {CLASS_NAMES[i]: round(class_weight_dict[i], 4) for i in class_weight_dict})

# 2. Build Base Model
print("\n[*] Initializing EfficientNetB0 backbone...")
base_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# 3. Extract Features for fast training
print("[*] Pre-extracting base features...")
feat_extractor = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()
])
t0 = time.time()
F_train = feat_extractor.predict(X_train, batch_size=32, verbose=1)
F_val = feat_extractor.predict(X_val, batch_size=32, verbose=1)
print(f"Feature extraction completed in {time.time()-t0:.2f}s. Feature shape: {F_train.shape}")

# 4. Train Classification Head
print("\n[*] Training Dense Classification Head...")
head_input = tf.keras.layers.Input(shape=(1280,))
x = tf.keras.layers.Dropout(0.3)(head_input)
head_output = tf.keras.layers.Dense(3, activation="softmax", name="predictions")(x)
head_model = tf.keras.Model(inputs=head_input, outputs=head_output)

head_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1)
]

history = head_model.fit(
    F_train, y_train,
    validation_data=(F_val, y_val),
    epochs=25,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# 5. Assemble Full End-to-End Model
print("\n[*] Assembling complete End-to-End EfficientNetB0 Model...")
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer_1")
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
# Reuse the trained dense head weights
outputs = head_model.layers[-1](x)

full_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="LungGuard_AI_EfficientNetB0")
full_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Evaluate on Test Dataset
print("\n[*] Evaluating on untouched test partition (N = 166)...")
test_loss, test_acc = full_model.evaluate(X_test, y_test, batch_size=16, verbose=1)
print(f"=== TEST ACCURACY: {test_acc*100:.2f}% | TEST LOSS: {test_loss:.4f} ===")

# Save Model
print(f"\n[*] Saving final verified model to: {TARGET_MODEL_PATH}")
os.makedirs(os.path.dirname(TARGET_MODEL_PATH), exist_ok=True)
full_model.save(TARGET_MODEL_PATH)
file_size_mb = os.path.getsize(TARGET_MODEL_PATH) / (1024 * 1024)
print(f"Model saved successfully! File size: {file_size_mb:.2f} MB")
print("=" * 65)
