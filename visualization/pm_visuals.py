import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model

# === Paths ===
dataset_base = r"E:/UG/B.E/Major Project/Code/Combined-Dataset"
real_dir = os.path.join(dataset_base, "real")
fake_dir = os.path.join(dataset_base, "fake")
output_root = r"E:/UG/B.E/Major Project/Code/Methodology_Visuals 6-6-2025"
os.makedirs(output_root, exist_ok=True)

# === Load MobileNet backbone
mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
feature_extractor = Model(inputs=mobilenet.input, outputs=mobilenet.get_layer("block_13_expand").output)
global_avg_model = Model(inputs=mobilenet.input, outputs=mobilenet.output)

# === Helpers
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def find_clearest_video(folder):
    best_score, best_path = -1, None
    for file in os.listdir(folder):
        if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
            path = os.path.join(folder, file)
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = variance_of_laplacian(gray)
                if score > best_score:
                    best_score = score
                    best_path = path
    return best_path

def extract_frames(video_path, num=10):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total // num, 1)
    frames = []
    for i in range(num):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def save_stack_image(frames, out_path, spacing=15):
    """
    Save a stacked overlapping image using OpenCV (no white padding).
    """
    h, w = 128, 128
    stack_width = w + (len(frames) - 1) * spacing
    canvas = np.ones((h, stack_width, 3), dtype=np.uint8) * 255

    for i, frame in enumerate(frames):
        img = cv2.resize(frame, (w, h))
        x = i * spacing
        canvas[:, x:x + w] = img

    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def preprocess_frames(frames, size=(128,128)):
    resized = [cv2.resize(f, size) for f in frames]
    arr = np.array(resized)
    return preprocess_input(arr)

def save_feature_maps(feature_map, path):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < feature_map.shape[-1]:
            ax.imshow(feature_map[0, :, :, i], cmap='viridis')
        ax.axis("off")
    plt.suptitle("Feature Maps (1st Frame)", fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_temporal_heatmap(features, path):
    t_feat = features.mean(axis=(1, 2))
    plt.figure(figsize=(10, 4))
    plt.imshow(t_feat.T, aspect="auto", cmap="plasma")
    plt.colorbar(label="Activation")
    plt.xlabel("Frame Index")
    plt.ylabel("Channels")
    plt.title("Temporal Feature Heatmap", fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_prediction_image(label, path):
    text = f"Final Prediction: {label}"
    
    # Estimate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Create image with some padding
    img = np.ones((200, text_width + 80, 3), dtype=np.uint8) * 255
    x_pos = (img.shape[1] - text_width) // 2
    y_pos = (img.shape[0] + text_height) // 2
    
    cv2.putText(img, text, (x_pos, y_pos), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


# === Main Pipeline
def run_pipeline(video_path, label, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"▶️ Processing {label} video: {video_path}")
    
    frames = extract_frames(video_path, num=5)

    if not frames:
        print("❌ No frames found.")
        return
    
    # === Save thumbnail with black-white film-strip effect ===
    thumb = frames[0].copy()
    thumb = cv2.resize(thumb, (128, 128))
    thumb_rgb = thumb.copy()

    # Create film strip canvas: black background, padding for sprockets
    strip_width = 148  # 10px padding each side
    strip_height = 148
    film_strip = np.zeros((strip_height, strip_width, 3), dtype=np.uint8)  # Black background

    # Place the thumbnail in center
    film_strip[10:138, 10:138] = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR)

    # Draw white sprocket holes (perforations)
    for y in range(15, strip_height - 15, 20):
        cv2.rectangle(film_strip, (2, y), (6, y + 8), (255, 255, 255), -1)              # Left side
        cv2.rectangle(film_strip, (strip_width - 6, y), (strip_width - 2, y + 8), (255, 255, 255), -1)  # Right side

    # Optional: draw triangle play icon in center (if needed)
    # pts = np.array([[74, 64], [74, 84], [90, 74]], np.int32)
    # cv2.fillPoly(film_strip, [pts], (255, 255, 255))

    # Save thumbnail
    cv2.imwrite(os.path.join(out_dir, "thumbnail.jpg"), film_strip)


    # Save stacked extracted frame image
    save_stack_image(frames, os.path.join(out_dir, "1_extracted_frames.jpg"), spacing=15)

    # Preprocessing
    preprocessed = preprocess_frames(frames)
    vis_pre = [((f + 1) * 127.5).astype(np.uint8) for f in preprocessed]

    save_stack_image(vis_pre, os.path.join(out_dir, "2_preprocessed_frames.jpg"), spacing=15)

    # Feature Maps
    fmap = feature_extractor.predict(preprocessed[:1])
    save_feature_maps(fmap, os.path.join(out_dir, "3_feature_maps.jpg"))

    # Temporal Features
    features = global_avg_model.predict(preprocessed)
    save_temporal_heatmap(features, os.path.join(out_dir, "4_temporal_heatmap.jpg"))

    # Final Prediction
    save_prediction_image(label, os.path.join(out_dir, "5_final_prediction.jpg"))

# === Run ===
real_video = find_clearest_video(real_dir)
fake_video = find_clearest_video(fake_dir)

run_pipeline(real_video, "Real", os.path.join(output_root, "Real"))
run_pipeline(fake_video, "Deep Fake", os.path.join(output_root, "Fake"))

print("\n✅ Paper-ready visuals saved in stacked layout!")
