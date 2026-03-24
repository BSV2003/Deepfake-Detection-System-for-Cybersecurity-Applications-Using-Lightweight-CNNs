import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten
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

print(f"✅ Loaded: {len(X_train)} train samples, {len(X_test)} test samples")

# Build MesoNet+LSTM
def build_mesonet_lstm(input_shape):
    def mesonet_block():
        model = Sequential()
        model.add(Conv2D(8, (3, 3), padding='same', input_shape=(128, 128, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(8, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(16, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(16, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))

        return model

    meso_base = mesonet_block()

    model = Sequential([
        TimeDistributed(meso_base, input_shape=input_shape),
        TimeDistributed(Flatten()),
        LSTM(64),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape = (frames_per_video, target_size[0], target_size[1], 3)
model = build_mesonet_lstm(input_shape)
model.summary()

callbacks = [
    ModelCheckpoint("E:/UG/B.E/Major Project/Code/Models/mesonet_lstm_baseline.keras", monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1)
]

print("🚀 Starting MesoNet+LSTM training...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=epochs, batch_size=batch_size, callbacks=callbacks)

# Final accuracies
final_train_acc = history.history['accuracy'][-1] * 100
final_val_acc = history.history['val_accuracy'][-1] * 100
print(f"\n✅ Final Training Accuracy: {final_train_acc:.2f}%")
print(f"✅ Final Validation Accuracy: {final_val_acc:.2f}%")
