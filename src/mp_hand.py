"""
MediaPipe Hand Landmarker (Tasks API) utilities.

This is the SOTA component of Assignment 2. We use Google's MediaPipe
Hand Landmarker (the current `mediapipe.tasks` API, model
`hand_landmarker.task`) to:

  1. detect the hand in each image,
  2. crop a tight, square region around the hand (removing most of the
     background), and
  3. optionally return the 21 normalized hand landmarks.

Cropping to the hand is what fixes the A1 weakness: in A1 the classifier
could cheat on the background, because every class was photographed in a
single short session (near-duplicate backgrounds). Classifying only the
hand region forces the model to rely on hand shape.

The module works with mediapipe >= 0.10 (Tasks API). It auto-downloads the
`hand_landmarker.task` model if it is not present.
"""

from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_TASK_PATH = "hand_landmarker.task"


def ensure_task_model(path: str = DEFAULT_TASK_PATH) -> str:
    """Download the hand_landmarker.task model if it is missing."""
    if not os.path.exists(path):
        print(f"[mp_hand] downloading hand_landmarker.task -> {path}")
        urllib.request.urlretrieve(TASK_URL, path)
    return path


def make_detector(task_path: str = DEFAULT_TASK_PATH,
                  min_conf: float = 0.3) -> mp_vision.HandLandmarker:
    """Create a single-hand landmark detector in image mode."""
    ensure_task_model(task_path)
    base_options = mp_python.BaseOptions(model_asset_path=task_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=min_conf,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def detect_landmarks(detector, rgb: np.ndarray):
    """Return (landmarks_px, landmarks_norm) or (None, None) if no hand.

    landmarks_px   : (21, 2) pixel coordinates in the input image.
    landmarks_norm : (21, 3) MediaPipe-normalized coordinates (x, y, z).
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None, None
    h, w = rgb.shape[:2]
    lm = result.hand_landmarks[0]
    px = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    norm = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    return px, norm


def hand_bbox(landmarks_px: np.ndarray, img_shape, margin: float = 0.35):
    """Square bounding box around the hand, expanded by `margin`, clamped."""
    h, w = img_shape[:2]
    x0, y0 = landmarks_px.min(axis=0)
    x1, y1 = landmarks_px.max(axis=0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * (1.0 + margin)
    side = max(side, 10.0)
    half = side / 2.0
    bx0 = int(max(0, cx - half))
    by0 = int(max(0, cy - half))
    bx1 = int(min(w, cx + half))
    by1 = int(min(h, cy + half))
    return bx0, by0, bx1, by1


def crop_hand(rgb: np.ndarray, detector, size: int = 128, margin: float = 0.35):
    """Return (crop_rgb, used_fallback, bbox, landmarks_norm).

    If a hand is found we crop around it; otherwise we fall back to a
    center square crop of the whole image so the pipeline never fails.
    """
    px, norm = detect_landmarks(detector, rgb)
    h, w = rgb.shape[:2]
    if px is None:
        # Fallback: center square crop of the full image.
        side = min(h, w)
        cy, cx = h // 2, w // 2
        bbox = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)
        crop = rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        crop = cv2.resize(crop, (size, size))
        return crop, True, bbox, None
    bbox = hand_bbox(px, rgb.shape, margin=margin)
    crop = rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    if crop.size == 0:
        crop = rgb
    crop = cv2.resize(crop, (size, size))
    return crop, False, bbox, norm


def normalized_landmark_vector(landmarks_norm: np.ndarray) -> np.ndarray:
    """Translation/scale invariant 63-d vector for a landmark classifier.

    Centered on the wrist (landmark 0) and scaled by the hand span, so it
    does not depend on where the hand is in the frame or its size.
    """
    if landmarks_norm is None:
        return np.zeros(63, dtype=np.float32)
    pts = landmarks_norm.copy()
    pts[:, :2] -= pts[0, :2]            # center on wrist (x, y)
    scale = np.linalg.norm(pts[:, :2], axis=1).max()
    if scale > 1e-6:
        pts[:, :2] /= scale
    return pts.reshape(-1).astype(np.float32)
