"""Отделение объекта от фона.

Два способа:
- простой — для однотонного фона: цвет фона берётся с краёв картинки, объект — всё, что заметно отличается,
  края уточняет GrabCut из OpenCV. Работает без видеокарты и без нейросетей.
- BiRefNet (MIT) — нейросеть для сложного фона: тени, отражения, пол, похожие цвета. Нужны PyTorch и
  transformers; модель загружается при первом вызове и остаётся в памяти.
"""

import importlib.util
import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Простой способ: цвет фона — медиана полосы по краю, отличие — расстояние в цветовом пространстве Lab.
BORDER_PX = 3
MIN_DIST = 12.0  # меньше этого — точно фон (шум, сжатие)
BORDER_SPREAD_FACTOR = 2.5  # порог не ниже 2.5 × разброса цвета самого фона
NONUNIFORM_SPREAD = 18.0  # разброс фона больше этого — фон неоднородный, простой способ ненадёжен
GRABCUT_MAX_EDGE = 800
GRABCUT_ITERATIONS = 4
MIN_PART_FRAC = 0.005  # куски меньше 0,5 % самого большого — мусор

# BiRefNet: официальная модель автора, версия закреплена — код модели скачивается с Hugging Face.
BIREFNET_REPO = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
BIREFNET_SIZE = 1024
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

METHODS = ("auto", "simple", "ai")

# Инструменты сервера работают в параллельных потоках. Нейросети (BiRefNet и SAM в parts.py) грузятся и считают
# по одному вызову за раз: иначе модель загрузится дважды, а видеопамять, нужная и Blender, уйдёт на два расчёта сразу.
GPU_LOCK = threading.Lock()


@dataclass
class Matte:
    alpha: np.ndarray  # float32, H×W, 0..1 — непрозрачность объекта
    method: str  # как получено, по-русски
    note: str = ""  # предупреждение для Claude, если результат может быть неточным

    @property
    def mask(self) -> np.ndarray:
        return self.alpha >= 0.5


def _border_pixels(a: np.ndarray, px: int = BORDER_PX) -> np.ndarray:
    return np.concatenate([a[:px].reshape(-1, a.shape[2]), a[-px:].reshape(-1, a.shape[2]), a[:, :px].reshape(-1, a.shape[2]), a[:, -px:].reshape(-1, a.shape[2])])


def _clean(mask: np.ndarray) -> np.ndarray:
    """Убирает мелкий мусор и закрывает щели в 1–2 px, не заливая настоящие дырки (кольцо остаётся кольцом)."""
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(count, bool)
    keep[1:] = areas >= MIN_PART_FRAC * areas.max()
    return keep[labels]


def simple_matte(img: Image.Image) -> Matte:
    rgb = np.asarray(img.convert("RGB"))
    height, width = rgb.shape[:2]
    scale = min(1.0, GRABCUT_MAX_EDGE / max(height, width))
    work = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else rgb

    lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB).astype(np.float32)
    background = np.median(_border_pixels(lab), axis=0)
    dist = np.linalg.norm(lab - background, axis=2)
    spread = float(np.percentile(np.linalg.norm(_border_pixels(lab) - background, axis=1), 90))
    threshold = max(MIN_DIST, BORDER_SPREAD_FACTOR * spread)

    if not (dist >= threshold).any():
        return Matte(np.zeros((height, width), np.float32), "простой (однотонный фон)", "Объект не найден: всё похоже на фон.")

    # GrabCut уточняет края по цветам объекта и фона; явно похожее на фон — точно фон, явно отличное — точно объект.
    gc = np.full(dist.shape, cv2.GC_PR_BGD, np.uint8)
    gc[dist < 0.5 * threshold] = cv2.GC_BGD
    gc[dist >= threshold] = cv2.GC_PR_FGD
    sure = cv2.erode((dist >= 3 * threshold).astype(np.uint8), np.ones((5, 5), np.uint8))
    gc[sure > 0] = cv2.GC_FGD
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(work, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK)
        mask = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    except cv2.error:  # GrabCut падает, если в одном из классов нет пикселей — тогда хватает порога
        mask = dist >= threshold
    mask = _clean(mask)

    alpha = mask.astype(np.float32)
    if scale < 1:
        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)  # мягкий край в 1 px для экспорта

    note = ""
    if spread > NONUNIFORM_SPREAD:
        note = "Фон неоднородный (пол, тени, градиент) — простой способ может ошибаться, лучше method='ai'."
    return Matte(alpha, "простой (однотонный фон)", note)


def ai_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


_model = None
_device = "cpu"


def _load_model():
    global _model, _device
    if _model is None:
        import torch
        from transformers import AutoModelForImageSegmentation

        log.info("Загружаю BiRefNet…")
        try:  # обычно модель уже на диске — в интернет не ходим
            model = AutoModelForImageSegmentation.from_pretrained(
                BIREFNET_REPO, revision=BIREFNET_REVISION, trust_remote_code=True, local_files_only=True
            )
        except OSError:  # первый запуск: скачать ~450 МБ
            model = AutoModelForImageSegmentation.from_pretrained(BIREFNET_REPO, revision=BIREFNET_REVISION, trust_remote_code=True)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(_device).eval()
        if _device == "cuda":
            model.half()
        _model = model
        log.info("BiRefNet загружен на %s", _device)
    return _model


def ai_matte(img: Image.Image) -> Matte:
    import torch

    rgb = img.convert("RGB")
    width, height = rgb.size
    x = np.asarray(rgb.resize((BIREFNET_SIZE, BIREFNET_SIZE), Image.Resampling.BILINEAR), np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    with GPU_LOCK:
        model = _load_model()
        tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(_device)
        if _device == "cuda":
            tensor = tensor.half()
        with torch.no_grad():
            pred = model(tensor)[-1].sigmoid()[0, 0].float().cpu().numpy()
        if _device == "cuda":
            torch.cuda.empty_cache()  # отдаём видеопамять обратно — она нужна Blender
    alpha = cv2.resize(pred, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(alpha, 0, 1).astype(np.float32)
    noise = ~_clean(alpha >= 0.5)  # мелкий мусор — в фон
    alpha[noise] = np.minimum(alpha[noise], 0.49)
    return Matte(alpha, f"BiRefNet ({'видеокарта' if _device == 'cuda' else 'процессор'})")


def warm_up() -> None:
    """Загружает BiRefNet заранее — сервер запускает это в фоне при старте, чтобы первый вызов не ждал загрузки."""
    if not ai_available():
        return
    try:
        with GPU_LOCK:
            _load_model()
        if _device == "cuda":
            ai_matte(Image.new("RGB", (64, 64)))  # первый расчёт на видеокарте медленный — пусть пройдёт сейчас
    except Exception:  # не вышло — модель загрузится при первом вызове, и там будет понятная ошибка
        log.exception("BiRefNet заранее не загрузился")


def matte(img: Image.Image, method: str = "auto") -> Matte:
    """method: "auto" — нейросеть, если установлена, иначе простой; "simple"; "ai"."""
    if method not in METHODS:
        raise ValueError(f"Способ должен быть одним из: {', '.join(METHODS)}")
    if method == "simple" or (method == "auto" and not ai_available()):
        result = simple_matte(img)
        if method == "auto" and result.note:
            result.note += " Нейросеть не установлена."
        return result
    if not ai_available():
        raise RuntimeError("Нейросеть BiRefNet не установлена (нужны torch и transformers). Используй method='simple'.")
    return ai_matte(img)
