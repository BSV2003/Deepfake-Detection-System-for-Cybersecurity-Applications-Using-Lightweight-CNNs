import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix
from tensorflow.keras.models import load_model
from video_data_generator import VideoDataGenerator
import seaborn as sns


# === Path Configurations ===
project_dir = "E:/UG/B.E/Major Project/Code"
result_dir = os.path.join(project_dir, "Results 5=10-6-2025")
os.makedirs(result_dir, exist_ok=True)

# Model paths
models = {
    "Baseline": {
        "MobileNet+LSTM": os.path.join(project_dir, "Models/mobilenet_lstm_baseline.keras"),
        "MesoNet+LSTM": os.path.join(project_dir, "Models/mesonet_lstm_baseline.keras")
    },
    "Tuned": {
        "MobileNet+LSTM": os.path.join(project_dir, "Models/mobilenet_lstm_tuned.keras"),
        "MesoNet+LSTM": os.path.join(project_dir, "Models/mesonet_lstm_tuned.keras")
    }
}

# Dataset paths
datasets = {
    "Celeb-DF": os.path.join(project_dir, "Video Datasets/Celeb-DF"),
    "DFDC": os.path.join(project_dir, "Video Datasets/DFDC"),
    "FF++": os.path.join(project_dir, "Video Datasets/FF++")
}

batch_size = {"Baseline": 8, "Tuned": 16}
target_size = (128, 128)
frames_per_video_map = {"Baseline": 5, "Tuned": 10}

results_all = {}
roc_data = {}

# Color mapping for ROC and accuracy plots
model_colors = {
    "Baseline-MobileNet+LSTM": "blue",
    "Tuned-MobileNet+LSTM": "red",
    "Baseline-MesoNet+LSTM": "green",
    "Tuned-MesoNet+LSTM": "darkorange"
}

for mode in models:
    results_all[mode] = []
    roc_data[mode] = {d: [] for d in datasets}

# === Plot Confusion Matrix ===
def plot_confusion_matrix(y_true, y_pred, model_name, dataset_name, mode):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 🔽 Print the values for reference
    print(f"📊 Confusion Matrix [{mode}] {model_name} on {dataset_name}:")
    print(f"    TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # 🔽 Plotting as before
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix\n{mode} - {model_name} - {dataset_name}")
    plt.tight_layout()
    path = os.path.join(result_dir, f"cm_{mode}_{model_name}_{dataset_name}.png")
    plt.savefig(path)
    plt.close()


# === Evaluate Model Function ===
def evaluate_model(model_path, model_name, dataset_path, dataset_name, mode):
    frames_per_video = frames_per_video_map[mode]
    print(f"\n📂 Evaluating [{mode}] {model_name} on {dataset_name}... Using {frames_per_video} frames per video.")

    generator = VideoDataGenerator(
        dataset_dir=dataset_path,
        batch_size=batch_size[mode],
        frames_per_video=frames_per_video,
        target_size=target_size,
        shuffle=False
    )

    total_videos = len(generator)
    total_frames_used = total_videos * frames_per_video
    print(f"🔢 Total videos: {total_videos}, Total frames used: {total_frames_used}")

    
    model = load_model(model_path)
    y_true, y_pred, y_probs = [], [], []

    for i in range(len(generator)):
        X_batch, y_batch = generator[i]
        probs = model.predict(X_batch).flatten()
        preds = (probs > 0.5).astype(int)
        y_true.extend(y_batch)
        y_pred.extend(preds)
        y_probs.extend(probs)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    plot_confusion_matrix(y_true, y_pred, model_name, dataset_name, mode)

    results_all[mode].append({
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": round(acc * 100, 2),
        "Precision": round(report["weighted avg"]["precision"], 2),
        "Recall": round(report["weighted avg"]["recall"], 2),
        "F1": round(report["weighted avg"]["f1-score"], 2)
    })

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    roc_data[mode][dataset_name].append({
        "model": model_name,
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc
    })


# === Run Evaluations ===
for mode in models:
    for model_name, model_path in models[mode].items():
        for dataset_name, dataset_path in datasets.items():
            evaluate_model(model_path, model_name, dataset_path, dataset_name, mode)


# === Export Tabulated Metrics ===
def export_combined_metrics():
    metrics_file = os.path.join(result_dir, "combined_model_metrics_5-6-2025.csv")
    with open(metrics_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Model", "Dataset", "Mode", "Accuracy", "Precision", "Recall", "F1"])
        for mode, results in results_all.items():
            for r in results:
                writer.writerow([
                    r['Model'], r['Dataset'], mode, r['Accuracy'],
                    r['Precision'], r['Recall'], r['F1']
                ])
    print(f"📄 Tabular metrics CSV saved at: {metrics_file}")

export_combined_metrics()


# === Combined ROC Plot ===
def plot_combined_roc():
    for dataset in datasets:
        plt.figure(figsize=(6, 6), dpi=300)
        for mode in roc_data:
            for entry in roc_data[mode][dataset]:
                label = f"{mode}-{entry['model']} (AUC = {entry['auc']:.2f})"
                color_key = f"{mode}-{entry['model']}"
                color = model_colors.get(color_key, "black")
                plt.plot(entry["fpr"], entry["tpr"], lw=2.5, label=label, color=color)

        plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
        plt.xlabel("False Positive Rate", fontsize=14, fontweight='bold')
        plt.ylabel("True Positive Rate", fontsize=14, fontweight='bold')
        plt.title(f"ROC Curves: {dataset}", fontsize=15, fontweight='bold')
        plt.gca().tick_params(axis='both', labelsize=12)
        plt.legend(loc="lower right", fontsize=10)

        for spine in plt.gca().spines.values():
            spine.set_linewidth(2)

        plt.tight_layout()
        jpeg_path = os.path.join(result_dir, f"roc_combined_{dataset}_5-6-2025.jpg")
        plt.savefig(jpeg_path, format='jpeg', dpi=300)
        plt.close()


plot_combined_roc()


# === Grouped Accuracy Bar Chart ===
def plot_grouped_accuracy_bar_from_csv():
    # Load the combined metrics from CSV
    csv_file_path = os.path.join(result_dir, "combined_model_metrics_5-6-2025.csv")
    df = pd.read_csv(csv_file_path)

    # Models to include (in desired display order)
    model_modes = ["Baseline-MobileNet+LSTM", "Tuned-MobileNet+LSTM",
                   "Baseline-MesoNet+LSTM", "Tuned-MesoNet+LSTM"]

    # Add a new column combining Mode and Model for plotting
    df["ModelMode"] = df["Mode"] + "-" + df["Model"]

    # Prepare data
    datasets_list = df["Dataset"].unique().tolist()
    accuracy_matrix = []

    for model_mode in model_modes:
        row = []
        for dataset in datasets_list:
            acc = df.loc[(df["ModelMode"] == model_mode) & (df["Dataset"] == dataset), "Accuracy"]
            row.append(acc.values[0] if not acc.empty else 0)
        accuracy_matrix.append(row)

    accuracy_matrix = np.array(accuracy_matrix)  # Shape: (4 models, 3 datasets)

    # Plotting
    x = np.arange(len(datasets_list))
    width = 0.2

    plt.figure(figsize=(12, 6), dpi=300)
    for i, model_mode in enumerate(model_modes):
        plt.bar(x + i * width, accuracy_matrix[i], width,
                label=model_mode, color=model_colors.get(model_mode, "gray"))

    plt.xlabel("Datasets", fontsize=13, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=13, fontweight='bold')
    plt.title("Model Accuracy across Datasets", fontsize=14, fontweight='bold')
    plt.xticks(x + 0.5 * width, datasets_list, fontsize=11)
    plt.yticks(fontsize=12)
    plt.ylim(0, 100)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "grouped_accuracy_bar_chart_from_csv.png"))
    plt.close()

    print("✅ Grouped accuracy bar chart plotted using CSV data.")


plot_grouped_accuracy_bar_from_csv()
