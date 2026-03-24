import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from video_data_generator import VideoDataGenerator
import os

# Dataset path
dataset_path = "E:/UG/B.E/Major Project/Code/Combined-Dataset"
os.makedirs("E:/UG/B.E/Major Project/Code/Models", exist_ok=True)

# Config
batch_size = 8
frames_per_video = 5
target_size = (128, 128)
epochs = 30

# Load data
print("🔄 Loading video data from generator...")
generator = VideoDataGenerator(
    dataset_dir=dataset_path,
    batch_size=batch_size,
    frames_per_video=frames_per_video,
    target_size=target_size,
    shuffle=True
)

split_idx = int(0.8 * len(generator))
X_train, y_train, X_test, y_test = [], [], [], []

for i in range(len(generator)):
    X_batch, y_batch = generator[i]
    if i < split_idx:
        X_train.append(X_batch)
        y_train.append(y_batch)
    else:
        X_test.append(X_batch)
        y_test.append(y_batch)

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)

print(f"✅ Data loaded: {len(X_train)} train videos, {len(X_test)} test videos")

# Build model
def build_mobilenet_lstm(input_shape):
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(128, 128, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = Sequential([
        TimeDistributed(base_model, input_shape=input_shape),
        TimeDistributed(GlobalAveragePooling2D()),
        LSTM(64),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape = (frames_per_video, target_size[0], target_size[1], 3)
model = build_mobilenet_lstm(input_shape)
model.summary()

callbacks = [
    ModelCheckpoint("E:/UG/B.E/Major Project/Code/Models/mobilenet_lstm_baseline.keras", monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1)
]

print("🚀 Starting MobileNet+LSTM training...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=epochs, batch_size=batch_size, callbacks=callbacks)

# Final accuracies
final_train_acc = history.history['accuracy'][-1] * 100
final_val_acc = history.history['val_accuracy'][-1] * 100
print(f"\n✅ Final Training Accuracy: {final_train_acc:.2f}%")
print(f"✅ Final Validation Accuracy: {final_val_acc:.2f}%")
