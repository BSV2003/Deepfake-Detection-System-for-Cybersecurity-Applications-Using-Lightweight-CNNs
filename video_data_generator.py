# video_data_generator.py

import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
import random

class VideoDataGenerator(Sequence):
    def __init__(self, dataset_dir, batch_size=8, frames_per_video=5, target_size=(128, 128), shuffle=True):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.frames_per_video = frames_per_video
        self.target_size = target_size
        self.shuffle = shuffle
        self.video_paths, self.labels = self._load_video_paths()
        self.on_epoch_end()

    def _load_video_paths(self):
        video_paths = []
        labels = []
        for label_name in ["real", "fake"]:
            label_dir = os.path.join(self.dataset_dir, label_name)
            for filename in os.listdir(label_dir):
                if filename.endswith(".mp4"):
                    video_paths.append(os.path.join(label_dir, filename))
                    labels.append(0 if label_name == "real" else 1)
        return video_paths, labels

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.video_paths, self.labels))
            random.shuffle(combined)
            self.video_paths, self.labels = zip(*combined)

    def __getitem__(self, idx):
        batch_paths = self.video_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_videos = [self._process_video(video_path) for video_path in batch_paths]
        return np.array(batch_videos), np.array(batch_labels)

    def _process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=np.int32)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.target_size)
                frame = frame.astype("float32") / 255.0
                frames.append(frame)
            else:
                break

        cap.release()

        while len(frames) < self.frames_per_video:
            frames.append(np.zeros((*self.target_size, 3), dtype=np.float32))

        return np.array(frames)
