from pathlib import Path
import sqlite3

import joblib
from fastapi import FastAPI, File, UploadFile

from feature_extractor import extract_features


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "training"
    / "models"
    / "image_quality_model.joblib"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "image_quality.db"
)


# ---------------------------------------------------------
# Load trained ML model
# ---------------------------------------------------------
model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
feature_columns = model_data["features"]


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------
app = FastAPI(
    title="AI-Powered Image Quality & Defect Detection",
    description="Image quality analysis API",
    version="1.0.0"
)


# ---------------------------------------------------------
# Database setup
# ---------------------------------------------------------
def init_database():
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            content_type TEXT,
            quality_score REAL,
            quality_label TEXT,
            confidence REAL,
            issues TEXT,
            width INTEGER,
            height INTEGER,
            sharpness REAL,
            brightness REAL,
            contrast REAL,
            saturation REAL,
            noise_level REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()
    connection.close()


init_database()


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# ---------------------------------------------------------
# Upload and analyze image
# ---------------------------------------------------------
@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    try:
        features = extract_features(image_bytes)

    except ValueError as e:
        return {
            "error": str(e)
        }

    # -----------------------------------------------------
    # Prepare features for ML model
    # -----------------------------------------------------
    feature_values = [
        features[column]
        for column in feature_columns
    ]

    # -----------------------------------------------------
    # ML prediction
    # -----------------------------------------------------
    prediction = model.predict(
        [feature_values]
    )[0]

    probabilities = model.predict_proba(
        [feature_values]
    )[0]

    confidence = float(
        max(probabilities)
    )

    # -----------------------------------------------------
    # Quality score
    # -----------------------------------------------------
    score_map = {
        "ACCEPTABLE": 90,
        "BLUR": 40,
        "UNDEREXPOSED": 45,
        "OVEREXPOSED": 45,
        "NOISY": 55
    }

    quality_score = score_map.get(
        prediction,
        50
    )

    # -----------------------------------------------------
    # Detected issues
    # -----------------------------------------------------
    issues = []

    if prediction == "BLUR":
        issues.append({
            "issue": "Blur",
            "severity": "HIGH",
            "confidence": round(
                confidence,
                4
            )
        })

    elif prediction == "UNDEREXPOSED":
        issues.append({
            "issue": "Underexposure",
            "severity": "MEDIUM",
            "confidence": round(
                confidence,
                4
            )
        })

    elif prediction == "OVEREXPOSED":
        issues.append({
            "issue": "Overexposure",
            "severity": "MEDIUM",
            "confidence": round(
                confidence,
                4
            )
        })

    elif prediction == "NOISY":
        issues.append({
            "issue": "Noise",
            "severity": "MEDIUM",
            "confidence": round(
                confidence,
                4
            )
        })

    # -----------------------------------------------------
    # Save analysis to SQLite
    # -----------------------------------------------------
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO analyses (
            filename,
            content_type,
            quality_score,
            quality_label,
            confidence,
            issues,
            width,
            height,
            sharpness,
            brightness,
            contrast,
            saturation,
            noise_level
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            file.filename,
            file.content_type,
            quality_score,
            prediction,
            confidence,
            str(issues),
            features["width"],
            features["height"],
            features["sharpness"],
            features["brightness"],
            features["contrast"],
            features["saturation"],
            features["noise_level"]
        )
    )

    connection.commit()

    analysis_id = cursor.lastrowid

    connection.close()

    # -----------------------------------------------------
    # Return analysis result
    # -----------------------------------------------------
    return {
        "analysis_id": analysis_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "quality_score": quality_score,
        "quality_label": prediction,
        "confidence": round(
            confidence,
            4
        ),
        "issues": issues,
        "features": features
    }


# ---------------------------------------------------------
# Analysis history
# ---------------------------------------------------------
@app.get("/analyses")
def get_analyses():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM analyses
        ORDER BY created_at DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return {
        "count": len(rows),
        "analyses": [
            dict(row)
            for row in rows
        ]
    }