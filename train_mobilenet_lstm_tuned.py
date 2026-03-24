import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from video_data_generator import VideoDataGenerator

# ✅ Dataset path and model save path
dataset_path = "E:/UG/B.E/Major Project/Code/Combined-Dataset"
model_save_path = "E:/UG/B.E/Major Project/Code/Models/mobilenet_lstm_tuned.keras"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# ✅ Tuned Config
batch_size = 16
frames_per_video = 10
target_size = (128, 128)
epochs = 50

# ✅ Load generator once
print("🔄 Preparing full video data generator...")
full_generator = VideoDataGenerator(
    dataset_dir=dataset_path,
    batch_size=batch_size,
    frames_per_video=frames_per_video,
    target_size=target_size,
    shuffle=True
)

# ✅ Smart train-test split using indexes
total_indexes = np.arange(len(full_generator.video_paths))
train_idx, val_idx = train_test_split(total_indexes, test_size=0.2, random_state=42, shuffle=True)

# ✅ Split into 2 new generators (manually assign paths + labels)
train_generator = VideoDataGenerator(
    dataset_dir=dataset_path,
    batch_size=batch_size,
    frames_per_video=frames_per_video,
    target_size=target_size,
    shuffle=True
)
train_generator.video_paths = [full_generator.video_paths[i] for i in train_idx]
train_generator.labels = [full_generator.labels[i] for i in train_idx]

val_generator = VideoDataGenerator(
    dataset_dir=dataset_path,
    batch_size=batch_size,
    frames_per_video=frames_per_video,
    target_size=target_size,
    shuffle=False
)
val_generator.video_paths = [full_generator.video_paths[i] for i in val_idx]
val_generator.labels = [full_generator.labels[i] for i in val_idx]

print(f"✅ Split done: {len(train_generator.video_paths)} training, {len(val_generator.video_paths)} validation")

# ✅ Build optimized MobileNet+LSTM model
def build_model(input_shape):
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(128, 128, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = Sequential([
        TimeDistributed(base_model, input_shape=input_shape),
        TimeDistributed(GlobalAveragePooling2D()),
        LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape = (frames_per_video, target_size[0], target_size[1], 3)
model = build_model(input_shape)
model.summary()

# ✅ Callbacks
callbacks = [
    ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1)
]

# ✅ Train
print("🚀 Starting optimized MobileNet+LSTM training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
    workers=4,                    # Use CPU parallelism
    use_multiprocessing=False     # Can be True if needed
)

# ✅ Final accuracy log
final_train_acc = history.history['accuracy'][-1] * 100
final_val_acc = history.history['val_accuracy'][-1] * 100
print(f"\n✅ Final Training Accuracy: {final_train_acc:.2f}%")
print(f"✅ Final Validation Accuracy: {final_val_acc:.2f}%")
