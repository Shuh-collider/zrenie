"""Режимы просмотра: обычный, контраст, контуры, тона."""

import cv2
import numpy as np
from PIL import Image

MODES = ("normal", "contrast", "edges", "tones")
MODE_NAMES = {"normal": "обычный", "contrast": "контраст", "edges": "контуры", "tones": "тона"}
# Чёрно-белые линии и плоские пятна — всегда PNG, JPEG их размывает.
LINE_MODES = {"edges", "tones"}

CLAHE_CLIP = 2.5
CLAHE_GRID = (8, 8)
# Контуры: порог по силе перепада — 80-й перцентиль среди заметных перепадов, не ниже 30.
EDGE_PERCENTILE = 80
EDGE_MIN_HIGH = 30.0
EDGE_LOW_RATIO = 0.4
# Под контурами — бледная копия картинки (15 %), чтобы было понятно, где что.
EDGE_GHOST = 0.15
TONE_LEVELS = 5


def _contrast(rgb: np.ndarray) -> np.ndarray:
    """Локальный контраст (CLAHE) по яркости: проявляет фаски, швы, рельеф в тенях и бликах."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _edges(gray: np.ndarray) -> np.ndarray:
    """Чёрные контуры на бледной подложке: силуэт и линии формы без цвета и бликов."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    magnitude = cv2.magnitude(cv2.Sobel(blurred, cv2.CV_32F, 1, 0), cv2.Sobel(blurred, cv2.CV_32F, 0, 1))
    significant = magnitude[magnitude > 8]
    high = float(np.percentile(significant, EDGE_PERCENTILE)) if significant.size else EDGE_MIN_HIGH
    high = max(high, EDGE_MIN_HIGH)
    edges = cv2.Canny(blurred, EDGE_LOW_RATIO * high, high, L2gradient=True)
    ghost = 255 - EDGE_GHOST * (255 - gray.astype(np.float32))
    return np.where(edges > 0, 0, ghost).astype(np.uint8)


def _tones(gray: np.ndarray) -> np.ndarray:
    """5 уровней яркости: видны плоскости света и тени — по ним легче читать объём."""
    lo, hi = np.percentile(gray, (1, 99))
    norm = np.clip((gray.astype(np.float32) - lo) / max(1.0, hi - lo), 0, 1)
    levels = np.minimum((norm * TONE_LEVELS).astype(np.int32), TONE_LEVELS - 1)
    return (levels * (255 / (TONE_LEVELS - 1))).astype(np.uint8)


def apply(img: Image.Image, mode: str) -> Image.Image:
    if mode == "normal":
        return img
    if mode not in MODES:
        raise ValueError(f"Неизвестный режим: {mode}")
    rgb = np.asarray(img.convert("RGB"))
    if mode == "contrast":
        return Image.fromarray(_contrast(rgb))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if mode == "edges":
        return Image.fromarray(_edges(gray), "L")
    return Image.fromarray(_tones(gray), "L")
