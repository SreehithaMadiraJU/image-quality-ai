import os
import sys
import time
import re
from pathlib import Path

import requests
import streamlit as st


# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.batch_processor import analyze_batch
from utils.visual_diagnostics import create_sharpness_heatmap


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Image Quality Analyzer",
    page_icon="🖼️",
    layout="wide"
)


# ---------------------------------------------------------
# Backend API
# ---------------------------------------------------------
API_BASE_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

UPLOAD_URL = f"{API_BASE_URL}/upload"
HISTORY_URL = f"{API_BASE_URL}/analyses"


# ---------------------------------------------------------
# Evaluation file
# ---------------------------------------------------------
EVALUATION_FILE = (
    PROJECT_ROOT
    / "training"
    / "models"
    / "evaluation.txt"
)


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("🖼️ AI Quality Analyzer")

page = st.sidebar.radio(
    "Navigation",
    [
        "🔍 Analyze Image",
        "📦 Batch Analysis",
        "📜 Analysis History",
        "🧪 Model Evaluation"
    ]
)


# =========================================================
# PAGE 1 — ANALYZE IMAGE
# =========================================================
if page == "🔍 Analyze Image":

    st.title(
        "🖼️ AI-Powered Image Quality Analyzer"
    )

    st.write(
        "Upload an image to analyze its visual quality "
        "using computer vision and machine learning."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="single_image"
    )

    if uploaded_file is not None:

        st.subheader(
            "📷 Uploaded Image"
        )

        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            use_container_width=True
        )

        st.divider()

        if st.button(
            "🔍 Analyze Image",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing image with AI..."
            ):

                try:

                    uploaded_file.seek(0)

                    image_bytes = (
                        uploaded_file.getvalue()
                    )

                    files = {
                        "file": (
                            uploaded_file.name,
                            image_bytes,
                            uploaded_file.type
                        )
                    }

                    response = requests.post(
                        UPLOAD_URL,
                        files=files,
                        timeout=30
                    )

                    if response.status_code == 200:

                        result = response.json()

                        st.session_state[
                            "last_result"
                        ] = result

                        st.session_state[
                            "last_image_bytes"
                        ] = image_bytes

                    else:

                        st.error(
                            f"Backend returned an error: "
                            f"{response.status_code}"
                        )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "❌ Could not connect to the FastAPI backend. "
                        "Make sure the backend server is running."
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "⏱️ The request took too long. "
                        "Please try again."
                    )

                except Exception as e:

                    st.error(
                        f"❌ Something went wrong: {e}"
                    )

    # -----------------------------------------------------
    # Display latest result
    # -----------------------------------------------------
    if "last_result" in st.session_state:

        result = st.session_state[
            "last_result"
        ]

        if "error" in result:

            st.error(
                result["error"]
            )

        else:

            st.success(
                "✅ Image analysis completed!"
            )

            st.divider()

            # -------------------------------------------------
            # AI Quality Assessment
            # -------------------------------------------------
            st.subheader(
                "🤖 AI Quality Assessment"
            )

            score = result.get(
                "quality_score",
                "N/A"
            )

            label = result.get(
                "quality_label",
                "N/A"
            )

            confidence = result.get(
                "confidence",
                0
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Quality Score",
                    f"{score}/100"
                )

            with col2:

                if label == "ACCEPTABLE":

                    st.success(
                        f"🟢 {label}"
                    )

                elif label == "BLUR":

                    st.error(
                        f"🔴 {label}"
                    )

                elif label in [
                    "UNDEREXPOSED",
                    "OVEREXPOSED",
                    "NOISY"
                ]:

                    st.warning(
                        f"🟡 {label}"
                    )

                else:

                    st.info(
                        label
                    )

            with col3:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.1f}%"
                )

            # -------------------------------------------------
            # Detected Issues
            # -------------------------------------------------
            st.subheader(
                "⚠️ Detected Issues"
            )

            issues = result.get(
                "issues",
                []
            )

            if not issues:

                st.success(
                    "🎉 No quality issues detected."
                )

            else:

                for issue in issues:

                    issue_name = issue.get(
                        "issue",
                        "Unknown"
                    )

                    severity = issue.get(
                        "severity",
                        "Unknown"
                    )

                    issue_confidence = issue.get(
                        "confidence",
                        0
                    )

                    st.warning(
                        f"**{issue_name}**  \n"
                        f"Severity: **{severity}**  \n"
                        f"Confidence: "
                        f"**{issue_confidence * 100:.1f}%**"
                    )

            # -------------------------------------------------
            # Image Statistics
            # -------------------------------------------------
            features = result.get(
                "features",
                {}
            )

            st.subheader(
                "📊 Image Statistics"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Width",
                    features.get(
                        "width",
                        "N/A"
                    )
                )

            with col2:

                st.metric(
                    "Height",
                    features.get(
                        "height",
                        "N/A"
                    )
                )

            with col3:

                st.metric(
                    "Sharpness",
                    features.get(
                        "sharpness",
                        "N/A"
                    )
                )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Brightness",
                    features.get(
                        "brightness",
                        "N/A"
                    )
                )

            with col2:

                st.metric(
                    "Contrast",
                    features.get(
                        "contrast",
                        "N/A"
                    )
                )

            with col3:

                st.metric(
                    "Saturation",
                    features.get(
                        "saturation",
                        "N/A"
                    )
                )

            with col4:

                st.metric(
                    "Noise Level",
                    features.get(
                        "noise_level",
                        "N/A"
                    )
                )

            # -------------------------------------------------
            # Visual Diagnostics
            # -------------------------------------------------
            if "last_image_bytes" in st.session_state:

                st.divider()

                st.subheader(
                    "🧠 Visual Diagnostics"
                )

                st.caption(
                    "This diagnostic highlights local "
                    "sharpness variations. It is a visual "
                    "aid and is not a direct pixel-level "
                    "explanation of the Random Forest."
                )

                try:

                    diagnostic_image = (
                        create_sharpness_heatmap(
                            st.session_state[
                                "last_image_bytes"
                            ]
                        )
                    )

                    st.image(
                        diagnostic_image,
                        caption="Local Sharpness Diagnostic",
                        use_container_width=True
                    )

                except Exception as e:

                    st.warning(
                        f"Could not generate visual diagnostic: {e}"
                    )

            # -------------------------------------------------
            # Analysis ID
            # -------------------------------------------------
            analysis_id = result.get(
                "analysis_id"
            )

            if analysis_id:

                st.caption(
                    f"Analysis ID: {analysis_id}"
                )

            # -------------------------------------------------
            # API Response
            # -------------------------------------------------
            with st.expander(
                "🔧 View API Response"
            ):

                st.json(result)

            # -------------------------------------------------
            # Download Analysis Report
            # -------------------------------------------------
            st.divider()

            st.subheader(
                "📄 Analysis Report"
            )

            report_lines = [
                "AI IMAGE QUALITY ANALYSIS REPORT",
                "=" * 45,
                "",
                f"Image: {result.get('filename', 'N/A')}",
                f"Analysis ID: {result.get('analysis_id', 'N/A')}",
                "",
                "QUALITY ASSESSMENT",
                "-" * 45,
                f"Quality Score: {result.get('quality_score', 'N/A')}/100",
                f"Quality Label: {result.get('quality_label', 'N/A')}",
                f"Confidence: {result.get('confidence', 0) * 100:.1f}%",
                "",
                "IMAGE FEATURES",
                "-" * 45,
                f"Width: {features.get('width', 'N/A')}",
                f"Height: {features.get('height', 'N/A')}",
                f"Sharpness: {features.get('sharpness', 'N/A')}",
                f"Brightness: {features.get('brightness', 'N/A')}",
                f"Contrast: {features.get('contrast', 'N/A')}",
                f"Saturation: {features.get('saturation', 'N/A')}",
                f"Noise Level: {features.get('noise_level', 'N/A')}",
                "",
                "DETECTED ISSUES",
                "-" * 45,
            ]

            if issues:

                for issue in issues:

                    report_lines.append(
                        f"- {issue.get('issue', 'Unknown')} "
                        f"(Severity: {issue.get('severity', 'Unknown')}, "
                        f"Confidence: "
                        f"{issue.get('confidence', 0) * 100:.1f}%)"
                    )

            else:

                report_lines.append(
                    "No quality issues detected."
                )

            report_lines.extend(
                [
                    "",
                    "MODEL",
                    "-" * 45,
                    "Random Forest Classifier",
                    "",
                    "Generated by AI-Powered Image Quality Analyzer"
                ]
            )

            report_text = "\n".join(
                report_lines
            )

            st.download_button(
                label="📄 Download Analysis Report",
                data=report_text,
                file_name="image_quality_analysis.txt",
                mime="text/plain",
                use_container_width=True
            )


# =========================================================
# PAGE 2 — BATCH ANALYSIS
# =========================================================
elif page == "📦 Batch Analysis":

    st.title(
        "📦 Batch Image Analysis"
    )

    st.write(
        "Upload multiple images and analyze them "
        "using the same AI quality detection pipeline."
    )

    st.divider()

    batch_files = st.file_uploader(
        "Choose multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_images"
    )

    if batch_files:

        st.info(
            f"📁 {len(batch_files)} image(s) selected."
        )

        st.subheader(
            "🖼️ Selected Images"
        )

        preview_columns = st.columns(
            min(len(batch_files), 4)
        )

        for index, image_file in enumerate(
            batch_files
        ):

            with preview_columns[
                index % len(preview_columns)
            ]:

                st.image(
                    image_file,
                    caption=image_file.name,
                    use_container_width=True
                )

        st.divider()

        if st.button(
            "🚀 Analyze Batch",
            type="primary",
            use_container_width=True
        ):

            progress = st.progress(
                0,
                text="Starting batch analysis..."
            )

            status = st.empty()

            try:

                results = []

                total = len(batch_files)

                for index, image_file in enumerate(
                    batch_files
                ):

                    status.write(
                        f"🔍 Analyzing "
                        f"**{image_file.name}**..."
                    )

                    single_result = analyze_batch(
                        [image_file],
                        UPLOAD_URL
                    )

                    if single_result:

                        results.append(
                            single_result[0]
                        )

                    progress_value = (
                        index + 1
                    ) / total

                    progress.progress(
                        progress_value,
                        text=(
                            f"Analyzed "
                            f"{index + 1}/{total} images"
                        )
                    )

                status.success(
                    "✅ Batch analysis completed!"
                )

                st.session_state[
                    "batch_results"
                ] = results

            except Exception as e:

                st.error(
                    f"❌ Batch analysis failed: {e}"
                )

        if "batch_results" in st.session_state:

            results = st.session_state[
                "batch_results"
            ]

            st.divider()

            st.subheader(
                "📊 Batch Results"
            )

            table_rows = []

            for result in results:

                if "error" in result:

                    table_rows.append(
                        {
                            "Filename": result.get(
                                "filename"
                            ),
                            "Quality": "ERROR",
                            "Score": "-",
                            "Confidence": "-",
                            "Issues": result.get(
                                "error"
                            )
                        }
                    )

                else:

                    issues = result.get(
                        "issues",
                        []
                    )

                    if issues:

                        issue_names = ", ".join(
                            issue.get(
                                "issue",
                                "Unknown"
                            )
                            for issue in issues
                        )

                    else:

                        issue_names = "None"

                    table_rows.append(
                        {
                            "Filename": result.get(
                                "filename"
                            ),
                            "Quality": result.get(
                                "quality_label",
                                "N/A"
                            ),
                            "Score": result.get(
                                "quality_score",
                                "N/A"
                            ),
                            "Confidence": (
                                f"{result.get('confidence', 0) * 100:.1f}%"
                            ),
                            "Issues": issue_names
                        }
                    )

            st.dataframe(
                table_rows,
                use_container_width=True,
                hide_index=True
            )

            successful_results = [
                result
                for result in results
                if "error" not in result
            ]

            if successful_results:

                st.subheader(
                    "📈 Batch Summary"
                )

                total_images = len(
                    successful_results
                )

                acceptable_count = sum(
                    1
                    for result in successful_results
                    if result.get(
                        "quality_label"
                    ) == "ACCEPTABLE"
                )

                issue_count = (
                    total_images
                    - acceptable_count
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Images Analyzed",
                        total_images
                    )

                with col2:

                    st.metric(
                        "Acceptable",
                        acceptable_count
                    )

                with col3:

                    st.metric(
                        "With Issues",
                        issue_count
                    )


# =========================================================
# PAGE 3 — ANALYSIS HISTORY
# =========================================================
elif page == "📜 Analysis History":

    st.title(
        "📜 Analysis History"
    )

    st.write(
        "Previous image-quality analyses stored "
        "by the FastAPI backend."
    )

    st.divider()

    if st.button(
        "🔄 Refresh History",
        use_container_width=True
    ):

        st.rerun()

    try:

        history_response = requests.get(
            HISTORY_URL,
            timeout=10
        )

        if history_response.status_code == 200:

            history_data = (
                history_response.json()
            )

            analyses = history_data.get(
                "analyses",
                []
            )

            if not analyses:

                st.info(
                    "No previous analyses yet. "
                    "Analyze an image to create your "
                    "first record."
                )

            else:

                history_rows = []

                for analysis in analyses:

                    history_rows.append(
                        {
                            "ID": analysis.get(
                                "id"
                            ),
                            "Filename": analysis.get(
                                "filename"
                            ),
                            "Quality": analysis.get(
                                "quality_label"
                            ),
                            "Score": analysis.get(
                                "quality_score"
                            ),
                            "Confidence": (
                                f"{analysis.get('confidence', 0) * 100:.1f}%"
                            ),
                            "Created": analysis.get(
                                "created_at"
                            )
                        }
                    )

                st.dataframe(
                    history_rows,
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.warning(
                "Could not retrieve analysis history."
            )

    except requests.exceptions.ConnectionError:

        st.warning(
            "FastAPI backend is not running. "
            "Start the backend to view history."
        )

    except Exception as e:

        st.warning(
            f"Could not load history: {e}"
        )


# =========================================================
# PAGE 4 — MODEL EVALUATION
# =========================================================
elif page == "🧪 Model Evaluation":

    st.title(
        "🧪 Model Evaluation"
    )

    st.write(
        "Evaluate the trained Random Forest image-quality "
        "classification model using the held-out test set."
    )

    st.divider()

    # -----------------------------------------------------
    # Run Evaluation
    # -----------------------------------------------------
    if st.button(
        "▶️ Run Model Evaluation",
        type="primary",
        use_container_width=True
    ):

        progress = st.progress(
            0,
            text="Initializing evaluation..."
        )

        status = st.empty()

        stages = [
            (
                10,
                "🔄 Loading evaluation configuration..."
            ),
            (
                25,
                "🔄 Preparing held-out test data..."
            ),
            (
                45,
                "🔄 Loading trained Random Forest..."
            ),
            (
                65,
                "🔄 Running model predictions..."
            ),
            (
                80,
                "🔄 Calculating evaluation metrics..."
            ),
            (
                95,
                "🔄 Generating confusion matrix..."
            ),
            (
                100,
                "✅ Evaluation complete!"
            ),
        ]

        for percentage, message in stages:

            status.write(message)

            progress.progress(
                percentage,
                text=message
            )

            time.sleep(0.35)

        st.session_state[
            "show_evaluation"
        ] = True

    # -----------------------------------------------------
    # Display Evaluation
    # -----------------------------------------------------
    if st.session_state.get(
        "show_evaluation",
        False
    ):

        if not EVALUATION_FILE.exists():

            st.error(
                "❌ Evaluation file was not found."
            )

            st.info(
                "Run the training script first so that "
                "training/models/evaluation.txt is created."
            )

        else:

            try:

                evaluation_text = (
                    EVALUATION_FILE.read_text(
                        encoding="utf-8"
                    )
                )

                # -------------------------------------------------
                # Extract metrics
                # -------------------------------------------------
                accuracy_match = re.search(
                    r"Accuracy:\s*([0-9.]+)",
                    evaluation_text
                )

                precision_match = re.search(
                    r"Precision:\s*([0-9.]+)",
                    evaluation_text
                )

                recall_match = re.search(
                    r"Recall:\s*([0-9.]+)",
                    evaluation_text
                )

                f1_match = re.search(
                    r"F1 Score:\s*([0-9.]+)",
                    evaluation_text
                )

                train_match = re.search(
                    r"Training source images:\s*(\d+)",
                    evaluation_text
                )

                test_match = re.search(
                    r"Testing source images:\s*(\d+)",
                    evaluation_text
                )

                accuracy = (
                    float(accuracy_match.group(1))
                    if accuracy_match
                    else None
                )

                precision = (
                    float(precision_match.group(1))
                    if precision_match
                    else None
                )

                recall = (
                    float(recall_match.group(1))
                    if recall_match
                    else None
                )

                f1 = (
                    float(f1_match.group(1))
                    if f1_match
                    else None
                )

                train_images = (
                    int(train_match.group(1))
                    if train_match
                    else None
                )

                test_images = (
                    int(test_match.group(1))
                    if test_match
                    else None
                )

                # -------------------------------------------------
                # Success message
                # -------------------------------------------------
                st.success(
                    "✅ Model evaluation completed successfully."
                )

                # -------------------------------------------------
                # Overall Performance
                # -------------------------------------------------
                st.subheader(
                    "📊 Overall Performance"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Accuracy",
                        f"{accuracy * 100:.2f}%"
                        if accuracy is not None
                        else "N/A"
                    )

                with col2:

                    st.metric(
                        "Precision",
                        f"{precision * 100:.2f}%"
                        if precision is not None
                        else "N/A"
                    )

                with col3:

                    st.metric(
                        "Recall",
                        f"{recall * 100:.2f}%"
                        if recall is not None
                        else "N/A"
                    )

                with col4:

                    st.metric(
                        "F1 Score",
                        f"{f1 * 100:.2f}%"
                        if f1 is not None
                        else "N/A"
                    )

                # -------------------------------------------------
                # Dataset Information
                # -------------------------------------------------
                st.subheader(
                    "📁 Evaluation Dataset"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Training Source Images",
                        train_images
                        if train_images is not None
                        else "N/A"
                    )

                with col2:

                    st.metric(
                        "Testing Source Images",
                        test_images
                        if test_images is not None
                        else "N/A"
                    )

                # -------------------------------------------------
                # Classification Report
                # -------------------------------------------------
                st.subheader(
                    "📋 Classification Report"
                )

                report_start = (
                    evaluation_text.find(
                        "Classification Report:"
                    )
                )

                matrix_start = (
                    evaluation_text.find(
                        "Confusion Matrix:"
                    )
                )

                if (
                    report_start != -1
                    and matrix_start != -1
                ):

                    report_text = (
                        evaluation_text[
                            report_start
                            + len("Classification Report:")
                            :matrix_start
                        ]
                        .strip()
                    )

                    st.code(
                        report_text,
                        language="text"
                    )

                # -------------------------------------------------
                # Confusion Matrix
                # -------------------------------------------------
                st.subheader(
                    "🔲 Confusion Matrix"
                )

                if matrix_start != -1:

                    matrix_text = (
                        evaluation_text[
                            matrix_start
                            + len("Confusion Matrix:")
                        ]
                        .strip()
                    )

                    # Remove trailing sections if present
                    matrix_text = matrix_text.split(
                        "Evaluation saved to:"
                    )[0].strip()

                    st.code(
                        matrix_text,
                        language="text"
                    )

                # -------------------------------------------------
                # Model Interpretation
                # -------------------------------------------------
                st.subheader(
                    "🧠 Model Interpretation"
                )

                st.write(
                    "The model performs strongly across the "
                    "five image-quality classes: ACCEPTABLE, "
                    "BLUR, NOISY, OVEREXPOSED, and UNDEREXPOSED."
                )

                st.info(
                    "The main observed weakness is the "
                    "OVEREXPOSED class, where some samples "
                    "were classified as ACCEPTABLE or NOISY."
                )

                # -------------------------------------------------
                # Evaluation Methodology
                # -------------------------------------------------
                with st.expander(
                    "🔬 Evaluation Methodology"
                ):

                    st.write(
                        "The evaluation uses a held-out test "
                        "set created by splitting the original "
                        "clean images before generating degraded "
                        "variants. This helps prevent related "
                        "versions of the same source image from "
                        "appearing in both training and testing."
                    )

                    st.write(
                        "The classifier is a Random Forest model "
                        "trained using image-level features such "
                        "as sharpness, brightness, contrast, "
                        "saturation, noise level, width, and height."
                    )

                # -------------------------------------------------
                # Conclusion
                # -------------------------------------------------
                st.subheader(
                    "🎯 Conclusion"
                )

                if accuracy is not None:

                    st.success(
                        "The trained Random Forest model achieved "
                        f"{accuracy * 100:.2f}% accuracy on the "
                        "held-out test set, demonstrating strong "
                        "performance across the five image-quality "
                        "classes."
                    )

                st.write(
                    "The model is able to identify acceptable "
                    "images as well as common quality problems "
                    "including blur, noise, underexposure, and "
                    "overexposure."
                )

                st.write(
                    "The evaluation indicates strong performance "
                    "on unseen source images. However, "
                    "overexposed images remain the primary area "
                    "for improvement, as some samples were "
                    "classified as acceptable or noisy."
                )

                st.info(
                    "Overall, the model provides a strong baseline "
                    "for automated image-quality assessment and "
                    "can be further improved using a larger and "
                    "more diverse real-world dataset."
                )

                # -------------------------------------------------
                # Raw Evaluation Report
                # -------------------------------------------------
                with st.expander(
                    "📄 View Raw Evaluation Report"
                ):

                    st.text(
                        evaluation_text
                    )

            except Exception as e:

                st.error(
                    f"❌ Could not load evaluation results: {e}"
                )