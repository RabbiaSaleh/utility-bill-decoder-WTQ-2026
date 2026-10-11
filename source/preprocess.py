"""
Image preprocessing for utility bills.
Handles tilt, low contrast, noise, and prepares the image for OCR / vision models.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load image as BGR numpy array."""
    img = cv2.imread(path)
    if img is None:
        # Fallback via PIL for some formats
        pil = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def deskew(image: np.ndarray) -> np.ndarray:
    """Estimate and correct skew angle using minAreaRect on text-like contours."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only correct meaningful skew
    if abs(angle) < 0.5 or abs(angle) > 15:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE on L channel for better text contrast."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(image: np.ndarray) -> np.ndarray:
    """Light bilateral filter to reduce noise while keeping edges."""
    return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)


def to_binary(image: np.ndarray) -> np.ndarray:
    """Adaptive threshold binary image useful for Tesseract."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return binary


def preprocess_for_ocr(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline.
    Returns (color_enhanced, binary) images.
    """
    img = load_image(path)
    img = deskew(img)
    img = enhance_contrast(img)
    img = denoise(img)
    binary = to_binary(img)
    return img, binary


def save_debug(image: np.ndarray, path: str) -> None:
    cv2.imwrite(path, image)