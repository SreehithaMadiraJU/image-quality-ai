import cv2
import numpy as np


def extract_features(image_bytes: bytes) -> dict:
    """
    Extract image-quality features from raw image bytes.
    """

    # Convert uploaded bytes into an OpenCV image
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to decode image")

    # Convert color spaces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. Sharpness
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Brightness
    brightness = float(np.mean(gray))

    # 3. Contrast
    contrast = float(np.std(gray))

    # 4. Saturation
    saturation = float(np.mean(hsv[:, :, 1]))

    # 5. High-frequency variation as a noise-related feature
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_map = cv2.absdiff(gray, blurred)
    noise_level = float(np.mean(noise_map))

    # Basic image information
    height, width = gray.shape

    return {
        "width": width,
        "height": height,
        "sharpness": round(float(sharpness), 4),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "saturation": round(saturation, 4),
        "noise_level": round(noise_level, 4),
    }