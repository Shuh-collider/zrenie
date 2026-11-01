"""Рисование поверх картинок: рамки кадров с номерами и лист из нескольких картинок."""

import logging
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .layout import Box

log = logging.getLogger(__name__)

# Шрифты с кириллицей: Windows, macOS (Pillow ищет имя файла с учётом регистра), Linux.
FONT_CANDIDATES = (
    "arial.ttf", "segoeui.ttf",
    "Arial.ttf", "Arial Unicode.ttf",
    "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "NotoSans-Regular.ttf", "FreeSans.ttf",
)
ACCENT = (255, 0, 255)  # пурпурный почти не встречается на референсах
SHEET_BACKGROUND = (60, 60, 60)
NEUTRAL_LIGHT = (220, 220, 220)  # бледный фон вокруг вырезанного объекта
SHEET_CELL = 640
SHEET_LABEL_H = 44
SHEET_GAP = 12


def _has_cyrillic(f: ImageFont.FreeTypeFont) -> bool:
    """Нет глифа — шрифт рисует тот же квадратик, что и для заведомо несуществующего символа."""
    return np.asarray(f.getmask("Ж")).tobytes() != np.asarray(f.getmask("\U0010ffff")).tobytes()


@lru_cache(maxsize=1)
def _font_path() -> str | None:
    for name in FONT_CANDIDATES:
        try:
            candidate = ImageFont.truetype(name, 20)
        except OSError:
            continue
        if _has_cyrillic(candidate):
            return candidate.path
    log.warning("Не найден шрифт с кириллицей — русские подписи на картинках будут квадратиками. Установите DejaVu Sans.")
    return None


@lru_cache(maxsize=8)
def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _font_path()
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)


def draw_boxes(img: Image.Image, boxes: list[Box], scale: float) -> Image.Image:
    """Рамки с номерами. boxes — в пикселях оригинала, scale — во сколько раз картинка меньше оригинала."""
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)
    width = max(2, round(min(out.size) / 300))
    label_font = font(max(14, round(min(out.size) / 28)))
    for number, box in enumerate(boxes, 1):
        x1, y1 = box.x1 / scale, box.y1 / scale
        x2, y2 = box.x2 / scale - 1, box.y2 / scale - 1
        draw.rectangle([x1, y1, x2, y2], outline=ACCENT, width=width)
        text = str(number)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=label_font)
        pad = width + 2
        draw.rectangle([x1, y1, x1 + right - left + 2 * pad, y1 + bottom - top + 2 * pad], fill=ACCENT)
        draw.text((x1 + pad - left, y1 + pad - top), text, fill="white", font=label_font)
    return out


@dataclass
class Placement:
    box: Box  # где картинка лежит на листе
    scale: float  # пикселей оригинала на пиксель листа


def compose_sheet(images: list[Image.Image], labels: list[str]) -> tuple[Image.Image, list[Placement]]:
    """Кладёт картинки в сетку одинаковых ячеек с подписями сверху."""
    n = len(images)
    rows = 1 if n <= 3 else 2 if n <= 6 else math.ceil(n / 3)
    cols = math.ceil(n / rows)
    cell_w = cell_h = SHEET_CELL
    sheet_w = cols * cell_w + (cols + 1) * SHEET_GAP
    sheet_h = rows * (cell_h + SHEET_LABEL_H) + (rows + 1) * SHEET_GAP
    sheet = Image.new("RGB", (sheet_w, sheet_h), SHEET_BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    label_font = font(26)

    placements = []
    for i, (img, label) in enumerate(zip(images, labels)):
        row, col = divmod(i, cols)
        cx = SHEET_GAP + col * (cell_w + SHEET_GAP)
        cy = SHEET_GAP + row * (cell_h + SHEET_LABEL_H + SHEET_GAP)
        draw.text((cx + 4, cy + SHEET_LABEL_H // 2), label, fill="white", font=label_font, anchor="lm")
        fit = min(cell_w / img.width, cell_h / img.height)
        w, h = max(1, round(img.width * fit)), max(1, round(img.height * fit))
        x = cx + (cell_w - w) // 2
        y = cy + SHEET_LABEL_H + (cell_h - h) // 2
        sheet.paste(img.convert("RGB").resize((w, h), Image.Resampling.LANCZOS), (x, y))
        placements.append(Placement(Box(x, y, x + w, y + h), img.width / w))
    return sheet, placements
