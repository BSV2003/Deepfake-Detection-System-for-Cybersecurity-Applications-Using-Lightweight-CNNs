import os
import shutil

# Original dataset paths
source_datasets = {
    "dfdc": "E:/UG/B.E/Major Project/Code/Video Datasets/DFDC",
    "ffpp": "E:/UG/B.E/Major Project/Code/Video Datasets/FF++",
    "celeb": "E:/UG/B.E/Major Project/Code/Video Datasets/Celeb-DF"
}

# Target combined dataset
combined_dataset_path = "E:/UG/B.E/Major Project/Code/Combined-Dataset"
os.makedirs(os.path.join(combined_dataset_path, "real"), exist_ok=True)
os.makedirs(os.path.join(combined_dataset_path, "fake"), exist_ok=True)

# Counter to ensure unique names
counter = {"real": 0, "fake": 0}

for name, path in source_datasets.items():
    for label in ["real", "fake"]:
        src_dir = os.path.join(path, label)
        dest_dir = os.path.join(combined_dataset_path, label)

        for file in os.listdir(src_dir):
            if file.endswith(".mp4"):
                src = os.path.join(src_dir, file)
                new_name = f"{name}_{label}_{counter[label]:04d}.mp4"
                dest = os.path.join(dest_dir, new_name)
                shutil.copyfile(src, dest)
                counter[label] += 1

print("✅ Datasets combined into 'Combined-Dataset/'")
