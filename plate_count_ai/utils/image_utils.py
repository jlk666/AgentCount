"""Shared image processing utilities for agar plate analysis."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image from path: {image_path}")
    return image


def laplacian_variance(image: np.ndarray) -> float:
    """Estimate blur using variance of Laplacian focus measure."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_stats(image: np.ndarray) -> dict[str, float | bool]:
    """Return simple brightness and contrast stats."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    return {
        "mean_brightness": mean,
        "std_brightness": std,
        "underexposed": mean < 45.0,
        "overexposed": mean > 210.0,
    }


def detect_plate_circle(image: np.ndarray) -> dict[str, Any]:
    """
    Detect plate geometry using Hough circles and contour fallback.

    Returns:
        {
          "found": bool,
          "center_x": int | None,
          "center_y": int | None,
          "radius": int | None,
          "score": float
        }
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(image.shape[:2]) // 2,
        param1=100,
        param2=30,
        minRadius=min(image.shape[:2]) // 6,
        maxRadius=min(image.shape[:2]) // 2,
    )

    if circles is not None and len(circles[0]) > 0:
        x, y, r = circles[0][0]
        return {
            "found": True,
            "center_x": int(x),
            "center_y": int(y),
            "radius": int(r),
            "score": 1.0,
        }

    # Fallback to largest contour approximated as circle-like.
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"found": False, "center_x": None, "center_y": None, "radius": None, "score": 0.0}

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, closed=True)
    if perimeter <= 0:
        return {"found": False, "center_x": None, "center_y": None, "radius": None, "score": 0.0}

    circularity = float((4 * np.pi * area) / (perimeter * perimeter))
    (x, y), radius = cv2.minEnclosingCircle(largest)
    found = circularity > 0.65 and radius > min(image.shape[:2]) * 0.2
    score = min(max(circularity, 0.0), 1.0)
    return {
        "found": bool(found),
        "center_x": int(x),
        "center_y": int(y),
        "radius": int(radius),
        "score": score,
    }


def crop_to_circle(image: np.ndarray, center_x: int, center_y: int, radius: int) -> np.ndarray:
    """Crop a square ROI around the detected plate circle."""
    h, w = image.shape[:2]
    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(w, center_x + radius)
    y2 = min(h, center_y + radius)
    return image[y1:y2, x1:x2].copy()
