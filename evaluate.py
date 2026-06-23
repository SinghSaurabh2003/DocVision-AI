import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.model_v2 import build_model

from src.config_v2 import (
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS
)

# =====================================================
# PATHS
# =====================================================

TEST_DATASET_PATH = "/kaggle/input/datasets/singhsaurabh03/testing/test"

CHECKPOINT_PATH = "/kaggle/input/datasets/singhsaurabh03/checkpoints/best_model.pth"

RESULTS_DIR = "/kaggle/working/Document-Type-Classifier/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print(f"Using Device : {device}")
print("=" * 60)

# =====================================================
# TRANSFORM
# =====================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# DATASET
# =====================================================

test_dataset = ImageFolder(
    TEST_DATASET_PATH,
    transform=transform
)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True

)

print(f"Total Test Images : {len(test_dataset)}")
print()

print("Classes")

for idx, cls in enumerate(test_dataset.classes):
    print(f"{idx} : {cls}")

print()

# =====================================================
# MODEL
# =====================================================

model = build_model().to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("=" * 60)
print("Checkpoint Loaded Successfully")
print("=" * 60)

print(f"Best Validation Accuracy : {checkpoint['best_accuracy']:.2f}%")

model.eval()

# =====================================================
# STORAGE
# =====================================================

all_predictions = []

all_labels = []

all_probabilities = []

image_paths = []

# =====================================================
# INFERENCE
# =====================================================

print("\n" + "=" * 60)
print("Running Evaluation...")
print("=" * 60)

with torch.no_grad():

    for images, labels in tqdm(test_loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predictions = torch.max(
            probabilities,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_probabilities.extend(
            confidence.cpu().numpy()
        )

# =====================================================
# DATAFRAME
# =====================================================

class_names = test_dataset.classes

prediction_df = pd.DataFrame({

    "Actual Label": [
        class_names[x]
        for x in all_labels
    ],

    "Predicted Label": [
        class_names[x]
        for x in all_predictions
    ],

    "Confidence": all_probabilities

})

prediction_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "predictions.csv"
    ),

    index=False

)

print()

print("Prediction CSV Saved")

print(
    os.path.join(
        RESULTS_DIR,
        "predictions.csv"
    )
)

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted"
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted"
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted"
)

print("\n")

print("=" * 60)
print("Evaluation Metrics")
print("=" * 60)

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(10, 8))

plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

plt.title("Confusion Matrix")

plt.colorbar()

tick_marks = np.arange(len(class_names))

plt.xticks(
    tick_marks,
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    tick_marks,
    class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > cm.max()/2 else "black"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()

print("Confusion Matrix Saved")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

report = classification_report(

    all_labels,

    all_predictions,

    target_names=class_names,

    digits=4

)

print("\n")

print("=" * 60)
print("Classification Report")
print("=" * 60)

print(report)

with open(

    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),

    "w"

) as f:

    f.write(report)

print("Classification Report Saved")

# =====================================================
# PER CLASS ACCURACY
# =====================================================

print("\n")

print("=" * 60)
print("Per Class Accuracy")
print("=" * 60)

per_class_results = []

for i, cls in enumerate(class_names):

    total = cm[i].sum()

    correct = cm[i][i]

    class_acc = (correct / total) * 100 if total != 0 else 0

    print(f"{cls:<20} : {class_acc:.2f}%")

    per_class_results.append(
        [cls, class_acc]
    )

per_class_df = pd.DataFrame(

    per_class_results,

    columns=[
        "Class",
        "Accuracy"
    ]

)

per_class_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "per_class_accuracy.csv"
    ),

    index=False

)

# =====================================================
# SAVE METRICS
# =====================================================

with open(

    os.path.join(
        RESULTS_DIR,
        "metrics.txt"
    ),

    "w"

) as f:

    f.write(f"Accuracy : {accuracy*100:.2f}%\n")

    f.write(f"Precision : {precision*100:.2f}%\n")

    f.write(f"Recall : {recall*100:.2f}%\n")

    f.write(f"F1 Score : {f1*100:.2f}%\n")

print()

print("Metrics Saved")

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n")

print("=" * 60)
print("Evaluation Completed Successfully")
print("=" * 60)

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")

print()

print("Results Saved In")

print(RESULTS_DIR)

print()

print("Generated Files")

print("-----------------------------")

print("confusion_matrix.png")

print("classification_report.txt")

print("metrics.txt")

print("predictions.csv")

print("per_class_accuracy.csv")

print("=" * 60)
