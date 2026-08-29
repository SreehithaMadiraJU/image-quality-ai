from pathlib import Path

from fastapi.testclient import TestClient

import sys

# Allow tests to import the backend application
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "backend" / "app"

sys.path.insert(0, str(APP_DIR))

from main import app


client = TestClient(app)


# ---------------------------------------------------------
# Test 1: Health check
# ---------------------------------------------------------
def test_health_check():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


# ---------------------------------------------------------
# Test 2: Upload a valid image
# ---------------------------------------------------------
def test_upload_image():

    image_path = (
        PROJECT_ROOT
        / "training"
        / "dataset"
        / "photos_no_class"
    )

    image_files = list(
        image_path.glob("*.jpg")
    )

    if not image_files:
        image_files = list(
            image_path.glob("*.jpeg")
        )

    if not image_files:
        image_files = list(
            image_path.glob("*.png")
        )

    assert image_files, "No test image found."

    test_image = image_files[0]

    with open(test_image, "rb") as image:

        response = client.post(
            "/upload",
            files={
                "file": (
                    test_image.name,
                    image,
                    "image/jpeg"
                )
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert "filename" in data
    assert "quality_score" in data
    assert "quality_label" in data
    assert "confidence" in data
    assert "issues" in data
    assert "features" in data


# ---------------------------------------------------------
# Test 3: Invalid image
# ---------------------------------------------------------
def test_invalid_image():

    response = client.post(
        "/upload",
        files={
            "file": (
                "invalid.txt",
                b"This is not an image.",
                "text/plain"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "error" in data


# ---------------------------------------------------------
# Test 4: Analysis history
# ---------------------------------------------------------
def test_analysis_history():

    response = client.get("/analyses")

    assert response.status_code == 200

    data = response.json()

    assert "count" in data
    assert "analyses" in data

    assert isinstance(
        data["analyses"],
        list
    )