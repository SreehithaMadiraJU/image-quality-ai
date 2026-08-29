import cv2
import numpy as np


def create_sharpness_heatmap(image_bytes: bytes):
    """
    Create a visual diagnostic map showing regions
    with relatively low or high local sharpness.

    This is a diagnostic visualization based on local
    image statistics. It is NOT a direct explanation
    of the Random Forest decision.
    """

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Unable to decode image"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Calculate local sharpness using Laplacian variance
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    local_sharpness = np.abs(
        laplacian
    )

    # Smooth the map so the visualization
    # represents broader regions
    heatmap = cv2.GaussianBlur(
        local_sharpness,
        (31, 31),
        0
    )

    # Normalize to 0-255
    heatmap = cv2.normalize(
        heatmap,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Apply OpenCV heatmap
    colored_heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Convert BGR → RGB for Streamlit
    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    original_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Blend original image with diagnostic map
    overlay = cv2.addWeighted(
        original_rgb,
        0.55,
        colored_heatmap,
        0.45,
        0
    )

    return overlay