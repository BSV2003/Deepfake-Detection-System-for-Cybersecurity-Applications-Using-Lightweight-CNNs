import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from video_data_generator import VideoDataGenerator

# ✅ Dataset path and model save path
dataset_path = "E:/UG/B.E/Major Project/Code/Combined-Dataset"
model_save_path = "E:/UG/B.E/Major Project/Code/Models/mesonet_lstm_tuned.keras"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# ✅ Tuned Configuration
batch_size = 16
frames_per_video = 10
target_size = (128, 128)
epochs = 50

# ✅ Load the full generator
print("🔄 Preparing video data generator...")
full_generator = VideoDataGenerator(
    dataset_dir=dataset_path,
    batch_size=batch_size,
    frames_per_video=frames_per_video,
    target_size=target_size,
    shuffle=True
)

# ✅ Train/Validation split using indexes
indexes = np.arange(len(full_generator.video_paths))
train_idx, val_idx = train_test_split(indexes, test_size=0.2, random_state=42, shuffle=True)

# ✅ Split into two generators
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

# ✅ Build tuned MesoNet+LSTM model
def build_model(input_shape):
    def mesonet_block():
        model = Sequential()
        model.add(Conv2D(16, (3, 3), padding='same', input_shape=(128, 128, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(32, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(32, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))

        return model

    meso_base = mesonet_block()

    model = Sequential([
        TimeDistributed(meso_base, input_shape=input_shape),
        TimeDistributed(Flatten()),
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

callbacks = [
    ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1)
]

# ✅ Train
print("🚀 Training MesoNet+LSTM (tuned)...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
    workers=4,
    use_multiprocessing=False
)

# ✅ Final Accuracy Report
final_train_acc = history.history['accuracy'][-1] * 100
final_val_acc = history.history['val_accuracy'][-1] * 100
print(f"\n✅ Final Training Accuracy: {final_train_acc:.2f}%")
print(f"✅ Final Validation Accuracy: {final_val_acc:.2f}%")
