# 🖼️ AI-Powered Image Quality & Defect Detection

An end-to-end computer vision and machine learning application that analyzes image quality and detects common visual defects such as blur, noise, underexposure, and overexposure.

The system combines OpenCV-based feature extraction, a Random Forest classifier, FastAPI, Streamlit, and automated testing into a complete AI application.

---

## ✨ Features

- 🖼️ Single image quality analysis
- 🤖 Random Forest based image-quality classification
- 📊 Quality score and prediction confidence
- ⚠️ Detection of image-quality issues
- 📦 Batch image analysis
- 🧠 Visual sharpness diagnostics
- 📜 Analysis history
- 🧪 Model evaluation dashboard
- 📄 Evaluation and analysis reports
- 🔌 FastAPI REST API
- 🧪 Pytest API testing
- 🌐 Interactive Streamlit interface

---

## 🧠 How It Works

The application extracts numerical image features using OpenCV and passes them to a trained Random Forest classifier.

Image Upload
    ↓
OpenCV Feature Extraction
    ↓
Image Features
    ↓
Random Forest Classifier
    ↓
Quality Prediction
    ↓
Quality Score + Confidence + Detected Issues

---

## 🔍 Image Quality Classes

| Class | Description |
|---|---|
| ACCEPTABLE | Image meets the quality requirements |
| BLUR | Image contains significant blur |
| NOISY | Image contains elevated noise |
| UNDEREXPOSED | Image is significantly too dark |
| OVEREXPOSED | Image is significantly too bright |

---

## 📊 Extracted Features

The model uses the following image-level features:

- Width
- Height
- Sharpness
- Brightness
- Contrast
- Saturation
- Noise Level

These features are extracted from the uploaded image using OpenCV.

---

## 📈 Model Performance

The Random Forest classifier was evaluated using a held-out test set.

| Metric | Score |
|---|---:|
| Accuracy | 97.50% |
| Precision | 97.65% |
| Recall | 97.50% |
| F1 Score | 97.45% |

### Classification Report

| Class | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| ACCEPTABLE | 0.94 | 1.00 | 0.97 |
| BLUR | 1.00 | 1.00 | 1.00 |
| NOISY | 0.94 | 1.00 | 0.97 |
| OVEREXPOSED | 1.00 | 0.88 | 0.93 |
| UNDEREXPOSED | 1.00 | 1.00 | 1.00 |

The main observed weakness is the OVEREXPOSED class, where a small number of samples were classified as ACCEPTABLE or NOISY.

The current training pipeline generates quality variations from clean source images. A larger and more diverse real-world dataset could further improve model robustness.

---

## 🏗️ System Architecture

Streamlit UI
    ↓
FastAPI REST API
    ↓
OpenCV Feature Extraction
    ↓
Random Forest Classifier
    ↓
Quality Prediction
    ↓
Analysis History

---

## 📦 Batch Image Analysis

The application supports analyzing multiple images at once.

For every uploaded image, the same quality-detection pipeline is applied.

The batch dashboard provides:

- Number of images analyzed
- Number of acceptable images
- Number of images with quality issues
- Individual analysis results
- Batch summary

---

## 🧪 Model Evaluation

The application includes a dedicated Model Evaluation page.

It displays:

- Accuracy
- Precision
- Recall
- F1 Score
- Training source images
- Testing source images
- Classification report
- Confusion matrix
- Model interpretation
- Evaluation conclusion

The evaluation page provides a clear view of how well the trained model performs on unseen test data.

---

## 🧠 Visual Diagnostics

The application includes a visual sharpness diagnostic.

This diagnostic highlights local variations in image sharpness and provides additional visual information about the image.

The visual diagnostic is a supporting visualization and is not a direct pixel-level explanation of the Random Forest model.

---

## 🔌 REST API

The FastAPI backend provides the following endpoints.

### Health Check

GET /health

Checks whether the backend is running.

### Upload Image

POST /upload

Uploads an image and returns its analysis.

The response includes:

- Analysis ID
- Filename
- Quality score
- Quality label
- Confidence
- Detected issues
- Extracted image features

### Analysis History

GET /analyses

Returns previously generated analyses.

### Interactive API Documentation

FastAPI automatically provides interactive API documentation at:

http://127.0.0.1:8000/docs

---

## 🧪 Testing

The API is tested using Pytest.

Current tests cover:

- Health check
- Image upload
- Invalid image handling
- Analysis history

Current test result:

4 passed

---

## 📁 Project Structure

image-quality-ai/
│
├── backend/
│   └── app/
│       ├── feature_extractor.py
│       └── main.py
│
├── frontend/
│   └── app.py
│
├── training/
│   ├── dataset/
│   │   └── photos_no_class/
│   │
│   ├── models/
│   │   ├── evaluation.txt
│   │   └── training_data.csv
│   │
│   └── train_model.py
│
├── tests/
│   └── test_api.py
│
├── utils/
│   ├── batch_processor.py
│   └── visual_diagnostics.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

The local virtual environment and training dataset are excluded from Git using .gitignore.

---

## 🔄 Machine Learning Training Pipeline

Clean Source Images
    ↓
Train / Test Source Split
    ↓
Generate Quality Variations
    ↓
Feature Extraction
    ↓
Random Forest Training
    ↓
Model Evaluation
    ↓
Save Model + Evaluation Results

For each clean source image, the training pipeline generates variations for:

- ACCEPTABLE
- BLUR
- UNDEREXPOSED
- OVEREXPOSED
- NOISY

The source images are split before generating variations to reduce the risk of related variations appearing in both training and testing sets.

---

## ⚙️ Installation

### 1. Clone the repository

git clone https://github.com/SreehithaMadiraJU/image-quality-ai.git

cd image-quality-ai

### 2. Create a virtual environment

Windows:

python -m venv backend/venv

Activate the environment:

backend\venv\Scripts\Activate.ps1

### 3. Install dependencies

pip install -r requirements.txt

---

## ▶️ Running the Backend

From the project root:

python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

The API will be available at:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs

---

## ▶️ Running the Frontend

Open a second terminal and activate the virtual environment.

Then:

cd frontend

streamlit run app.py

The Streamlit application will open in your browser.

---

## 🧑‍💻 Training the Model

To retrain the Random Forest model:

cd training

python train_model.py

The training script:

1. Loads the clean source images.
2. Generates quality variations.
3. Extracts image features.
4. Trains the Random Forest classifier.
5. Evaluates the model.
6. Saves the training data and evaluation results.

---

## ⚠️ Limitations

The current system has several limitations:

- The training variations are programmatically generated.
- The model relies on handcrafted image-level features.
- The classifier supports five predefined quality categories.
- Real-world image defects may differ from the generated training variations.
- A larger real-world dataset could improve generalization.
- The visual diagnostic is not a direct explanation of the Random Forest decision.

---

## 🔮 Future Improvements

Possible future improvements include:

- Larger real-world image-quality datasets
- Deep-learning-based image-quality assessment
- Additional defect categories
- Object-aware quality detection
- Advanced model explainability
- More detailed analytics
- Automated model retraining
- Cloud deployment
- User authentication

---

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Image processing and feature extraction |
| NumPy | Numerical computation |
| Pandas | Dataset processing |
| Scikit-learn | Machine learning |
| Random Forest | Image-quality classification |
| FastAPI | REST API |
| Uvicorn | ASGI server |
| Streamlit | Web interface |
| SQLite | Analysis history |
| Pytest | Automated testing |
| Joblib | Model serialization |

---

## 📜 License

This project is licensed under the MIT License.

See the LICENSE file for details.

---

## 👩‍💻 Author

Sreehitha Madiraju

Computer Science (AI/ML)

GitHub:
https://github.com/SreehithaMadiraJU

---

## ⭐ Project Highlights

AI + Computer Vision + Machine Learning + REST API + Web UI + Testing

This project demonstrates an end-to-end machine learning application that takes an image as input, extracts visual features, classifies its quality, and presents the results through an interactive web interface.

### Key Result

97.50% Accuracy on the held-out evaluation set
