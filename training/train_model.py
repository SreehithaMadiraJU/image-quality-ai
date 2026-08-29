from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "training"
    / "dataset"
    / "photos_no_class"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "training"
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Import feature extractor
# ---------------------------------------------------------
APP_DIR = PROJECT_ROOT / "backend" / "app"

sys.path.insert(
    0,
    str(APP_DIR)
)

from feature_extractor import extract_features


# ---------------------------------------------------------
# Features used by the model
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    "width",
    "height",
    "sharpness",
    "brightness",
    "contrast",
    "saturation",
    "noise_level",
]


# ---------------------------------------------------------
# Image degradation functions
# ---------------------------------------------------------
def make_blur(image):

    return cv2.GaussianBlur(
        image,
        (15, 15),
        0
    )


def make_underexposed(image):

    return np.clip(
        image.astype(np.float32) * 0.35,
        0,
        255
    ).astype(np.uint8)


def make_overexposed(image):

    return np.clip(
        image.astype(np.float32) * 1.8,
        0,
        255
    ).astype(np.uint8)


def make_noisy(image):

    noise = np.random.normal(
        0,
        25,
        image.shape
    )

    noisy = (
        image.astype(np.float32)
        + noise
    )

    return np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)


# ---------------------------------------------------------
# Extract features from OpenCV image
# ---------------------------------------------------------
def get_features(image):

    success, encoded = cv2.imencode(
        ".jpg",
        image
    )

    if not success:
        raise ValueError(
            "Could not encode image"
        )

    return extract_features(
        encoded.tobytes()
    )


# ---------------------------------------------------------
# Find original images
# ---------------------------------------------------------
def find_images():

    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]:

        image_files.extend(
            DATASET_DIR.glob(extension)
        )

    if not image_files:

        raise FileNotFoundError(
            "No images found in dataset folder."
        )

    return sorted(image_files)


# ---------------------------------------------------------
# Create samples from images
# ---------------------------------------------------------
def create_samples(image_files):

    rows = []

    for image_path in image_files:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"Skipping: {image_path.name}"
            )

            continue

        variations = [
            (
                "ACCEPTABLE",
                image
            ),
            (
                "BLUR",
                make_blur(image)
            ),
            (
                "UNDEREXPOSED",
                make_underexposed(image)
            ),
            (
                "OVEREXPOSED",
                make_overexposed(image)
            ),
            (
                "NOISY",
                make_noisy(image)
            ),
        ]

        for label, variant in variations:

            features = get_features(
                variant
            )

            rows.append({
                **features,
                "label": label
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------
def train():

    print("=" * 60)
    print("AI IMAGE QUALITY MODEL TRAINING")
    print("=" * 60)

    image_files = find_images()

    print()
    print(
        f"Found {len(image_files)} clean images."
    )

    # -----------------------------------------------------
    # IMPORTANT:
    # Split ORIGINAL images before generating variations.
    # -----------------------------------------------------
    train_files, test_files = train_test_split(
        image_files,
        test_size=0.20,
        random_state=42
    )

    print()
    print(
        f"Training source images: {len(train_files)}"
    )

    print(
        f"Testing source images:  {len(test_files)}"
    )

    # -----------------------------------------------------
    # Generate training samples
    # -----------------------------------------------------
    print()
    print("Generating training samples...")

    train_dataframe = create_samples(
        train_files
    )

    # -----------------------------------------------------
    # Generate testing samples
    # -----------------------------------------------------
    print("Generating testing samples...")

    test_dataframe = create_samples(
        test_files
    )

    print()
    print(
        f"Training samples: {len(train_dataframe)}"
    )

    print(
        f"Testing samples: {len(test_dataframe)}"
    )

    # -----------------------------------------------------
    # Save generated dataset
    # -----------------------------------------------------
    training_data_file = (
        MODEL_DIR
        / "training_data.csv"
    )

    train_dataframe.to_csv(
        training_data_file,
        index=False
    )

    print()
    print(
        f"Saved training data to: "
        f"{training_data_file}"
    )

    # -----------------------------------------------------
    # Prepare X and y
    # -----------------------------------------------------
    X_train = train_dataframe[
        FEATURE_COLUMNS
    ]

    y_train = train_dataframe[
        "label"
    ]

    X_test = test_dataframe[
        FEATURE_COLUMNS
    ]

    y_test = test_dataframe[
        "label"
    ]

    # -----------------------------------------------------
    # Train Random Forest
    # -----------------------------------------------------
    print()
    print("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------
    predictions = model.predict(
        X_test
    )

    # -----------------------------------------------------
    # Evaluation metrics
    # -----------------------------------------------------
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )

    # -----------------------------------------------------
    # Display evaluation
    # -----------------------------------------------------
    print()
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print()
    print("Classification Report:")
    print(report)

    print()
    print("Confusion Matrix:")
    print(matrix)

    # -----------------------------------------------------
    # Save evaluation results
    # -----------------------------------------------------
    evaluation_file = (
        MODEL_DIR
        / "evaluation.txt"
    )

    with open(
        evaluation_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "AI IMAGE QUALITY MODEL EVALUATION\n"
        )

        file.write(
            "=" * 50 + "\n\n"
        )

        file.write(
            f"Training source images: "
            f"{len(train_files)}\n"
        )

        file.write(
            f"Testing source images: "
            f"{len(test_files)}\n\n"
        )

        file.write(
            f"Accuracy:  {accuracy:.4f}\n"
        )

        file.write(
            f"Precision: {precision:.4f}\n"
        )

        file.write(
            f"Recall:    {recall:.4f}\n"
        )

        file.write(
            f"F1 Score:  {f1:.4f}\n\n"
        )

        file.write(
            "Classification Report:\n"
        )

        file.write(
            report
        )

        file.write(
            "\nConfusion Matrix:\n"
        )

        file.write(
            np.array2string(matrix)
        )

    # -----------------------------------------------------
    # Save trained model
    # -----------------------------------------------------
    model_file = (
        MODEL_DIR
        / "image_quality_model.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "classes": list(
                model.classes_
            )
        },
        model_file
    )

    print()
    print(
        f"Evaluation saved to: "
        f"{evaluation_file}"
    )

    print()
    print("=" * 60)
    print(
        f"MODEL SAVED: {model_file}"
    )
    print("=" * 60)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    train()