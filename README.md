\# 🖼️ AI-Powered Image Quality \& Defect Detection



An end-to-end AI-powered image quality analysis system that uses computer vision and machine learning to automatically identify common image-quality problems such as blur, noise, underexposure, and overexposure.



The project combines OpenCV-based feature extraction, a Random Forest classifier, a FastAPI REST API, SQLite analysis history, and a Streamlit web interface.



\---



\## 🚀 Features



\- 🖼️ Single-image quality analysis

\- 🤖 Machine-learning-based quality classification

\- 📊 Quality score and prediction confidence

\- ⚠️ Detection of common image-quality issues

\- 📦 Batch image analysis

\- 🧠 Visual sharpness diagnostics

\- 📜 Analysis history using SQLite

\- 🧪 Model evaluation dashboard

\- 📄 Downloadable analysis reports

\- 🔌 REST API using FastAPI

\- 🧪 Automated API tests using Pytest

\- 🌐 Streamlit-based user interface



\---



\## 🧠 Machine Learning



The system uses a \*\*Random Forest Classifier\*\* to classify images into five quality categories:



| Class | Description |

|---|---|

| `ACCEPTABLE` | Image meets the quality baseline |

| `BLUR` | Image contains significant blur |

| `NOISY` | Image contains elevated noise |

| `UNDEREXPOSED` | Image is significantly too dark |

| `OVEREXPOSED` | Image is significantly too bright |



\### Feature Extraction



The model uses image-level features extracted using OpenCV:



\- Width

\- Height

\- Sharpness

\- Brightness

\- Contrast

\- Saturation

\- Noise level



\---



\## 📊 Model Performance



The trained Random Forest model was evaluated on a held-out test set.



| Metric | Result |

|---|---:|

| Accuracy | \*\*97.50%\*\* |

| Precision | \*\*97.65%\*\* |

| Recall | \*\*97.50%\*\* |

| F1 Score | \*\*97.45%\*\* |



\### Classification Performance



The model performs strongly across all five image-quality classes.



The main observed weakness is the `OVEREXPOSED` class, where a small number of samples were classified as `ACCEPTABLE` or `NOISY`.



> The current dataset uses clean source images and programmatically generated quality variations. A larger and more diverse real-world dataset would help improve robustness.



\---



\## 🏗️ System Architecture



```text

&#x20;                   ┌─────────────────────┐

&#x20;                   │   Streamlit UI      │

&#x20;                   │                     │

&#x20;                   │ Single / Batch /    │

&#x20;                   │ History / Evaluation│

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              │ HTTP

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │    FastAPI API      │

&#x20;                   │                     │

&#x20;                   │ Image Upload        │

&#x20;                   │ Prediction          │

&#x20;                   │ Analysis History    │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                ┌─────────────┴─────────────┐

&#x20;                ▼                           ▼

&#x20;      ┌──────────────────┐        ┌─────────────────┐

&#x20;      │ OpenCV Feature   │        │ Random Forest   │

&#x20;      │ Extraction       │───────▶│ Classifier      │

&#x20;      └──────────────────┘        └─────────────────┘

&#x20;                                          │

&#x20;                                          ▼

&#x20;                                 Quality Prediction

&#x20;                                          │

&#x20;                                          ▼

&#x20;                                 ┌─────────────────┐

&#x20;                                 │ SQLite History  │

&#x20;                                 └─────────────────┘

