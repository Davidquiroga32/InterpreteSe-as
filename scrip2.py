import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ======== Modelo Transformer ========
def build_transformer_model(seq_len, feat_dim, num_classes,
                            d_model=128, num_heads=4, ff_dim=256, num_layers=2):
    inputs = layers.Input(shape=(seq_len, feat_dim))
    x = layers.Dense(d_model)(inputs)

    # Positional encoding con Embedding
    pos = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=d_model)(pos)
    x = x + pos_emb

    for _ in range(num_layers):
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        ff = layers.Dense(ff_dim, activation="relu")(x)
        ff = layers.Dense(d_model)(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ======== Cargar dataset ========
DATA_DIR = "dataset"
CLASS_NAMES = sorted(os.listdir(DATA_DIR))
SEQ_LEN = 32
FEAT_DIM = 84

X, y = [], []
for idx, cname in enumerate(CLASS_NAMES):
    class_dir = os.path.join(DATA_DIR, cname)
    for f in os.listdir(class_dir):
        path = os.path.join(class_dir, f)
        arr = np.load(path).astype(np.float32)  # (32,84)
        X.append(arr)
        y.append(idx)

X = np.array(X)
y = np.array(y)
print("Dataset:", X.shape, y.shape)

# Shuffle
idxs = np.arange(len(X))
np.random.shuffle(idxs)
X, y = X[idxs], y[idxs]

# Split train/val
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ======== Entrenar ========
model = build_transformer_model(SEQ_LEN, FEAT_DIM, num_classes=len(CLASS_NAMES))
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16
)

# ======== Guardar ========
model.save("modelo_signos.h5")
print("Modelo guardado en modelo_signos.h5")
