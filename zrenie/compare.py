"""Сравнение рендера модели с референсом: выравнивание силуэтов, совпадение (IoU), где и насколько расходятся."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import draw
from .layout import Box
from .matting import Matte
from .views import Shape, measure

BANDS = 10  # полос по высоте для отчёта «где шире / уже»
BAND_REPORT_DIFF = 0.03  # разницу ширины меньше 3 % не сообщаем
EDGE_REPORT_SHIFT = 0.02  # сдвиг края меньше 2 % ширины объекта не сообщаем
BLOB_MIN_FRAC = 0.004  # пятна «не хватает / лишнее» меньше 0,4 % площади — шум по контуру
BLOB_REPORT = 3
KEY_MATCH_TOL = 0.08
KEY_REPORT_DIFF = 0.015

MISSING = (230, 40, 40)  # есть на референсе, нет у модели
EXTRA = (40, 110, 255)  # есть у модели, нет на референсе
REF_EDGE = (0, 190, 0)
MODEL_EDGE = (255, 0, 255)
PANEL_HEIGHT = 900
PANEL_GAP = 24
LABEL_H = 46
BACKGROUND = (60, 60, 60)


def alpha_matte(img: Image.Image) -> Matte | None:
    """Силуэт по прозрачности рендера, если фон действительно прозрачный."""
    if img.mode not in ("RGBA", "LA", "PA") and not (img.mode == "P" and "transparency" in img.info):
        return None
    alpha = np.asarray(img.convert("RGBA"))[..., 3].astype(np.float32) / 255
    if alpha.min() > 0.98:  # прозрачности нет — фон нарисован
        return None
    return Matte(alpha, "альфа-канал рендера (точно)")


@dataclass
class Comparison:
    iou: float
    scale: float  # во сколько раз увеличен рендер, чтобы совпасть по высоте
    ref: Shape
    model: Shape  # силуэт модели после выравнивания, в координатах референса
    ref_mask: np.ndarray
    model_mask: np.ndarray  # в координатах референса
    model_rgb: np.ndarray  # рендер в координатах референса (для наложения)
    findings: list[str]
    matrix: np.ndarray  # 2×3: пиксели рендера → пиксели референса (для деталей модели)


def _extent(mask: np.ndarray, y1: int, y2: int) -> tuple[int, int] | None:
    cols = np.nonzero(mask[y1:y2].any(axis=0))[0]
    if len(cols) == 0:
        return None
    return int(cols[0]), int(cols[-1]) + 1


def _where(box: Box, ref: Box) -> str:
    """Положение пятна словами и в процентах рамки референса."""
    x1 = (box.x1 - ref.x1) / ref.w * 100
    x2 = (box.x2 - ref.x1) / ref.w * 100
    z1 = (ref.y2 - box.y2) / ref.h * 100
    z2 = (ref.y2 - box.y1) / ref.h * 100
    cx, cz = (x1 + x2) / 2, (z1 + z2) / 2
    horizontal = "слева" if cx < 33 else "справа" if cx > 67 else "по центру"
    vertical = "внизу" if cz < 33 else "вверху" if cz > 67 else "посередине"
    return f"{vertical} {horizontal} (x {x1:.0f}–{x2:.0f} %, высота {z1:.0f}–{z2:.0f} %)"


def _blobs(mask: np.ndarray, ref: Box, area: float) -> list[tuple[float, Box]]:
    clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    blobs = []
    for i in range(1, count):
        a = stats[i, cv2.CC_STAT_AREA]
        if a >= BLOB_MIN_FRAC * area:
            x, y, w, h = stats[i, :4]
            blobs.append((a / area, Box(int(x), int(y), int(x + w), int(y + h))))
    return sorted(blobs, key=lambda b: -b[0])


def compare(ref_img: Image.Image, ref_matte: Matte, render_img: Image.Image, render_matte: Matte) -> Comparison | None:
    ref = measure(ref_matte)
    model0 = measure(render_matte)
    if ref is None or model0 is None:
        return None
    height, width = ref_matte.mask.shape
    # Выравниваем по высоте и низу: рендер масштабируется так, чтобы высота силуэта совпала, низ и центр рамки — тоже.
    s = ref.bbox.h / model0.bbox.h
    m_cx, r_cx = (model0.bbox.x1 + model0.bbox.x2) / 2, (ref.bbox.x1 + ref.bbox.x2) / 2
    matrix = np.float32([[s, 0, r_cx - m_cx * s], [0, s, ref.bbox.y2 - model0.bbox.y2 * s]])
    warped_alpha = cv2.warpAffine(render_matte.alpha, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)
    model_mask = warped_alpha >= 0.5
    model_rgb = cv2.warpAffine(np.asarray(render_img.convert("RGB")), matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
    model = measure(Matte(warped_alpha, render_matte.method))
    ref_mask = ref_matte.mask
    union = np.logical_or(ref_mask, model_mask).sum()
    iou = float(np.logical_and(ref_mask, model_mask).sum() / union) if union else 0.0

    findings: list[str] = []
    rb = ref.bbox
    aspect_diff = (model.aspect - ref.aspect) / ref.aspect * 100
    if abs(aspect_diff) >= BAND_REPORT_DIFF * 100:
        findings.append(f"В целом модель {'шире' if aspect_diff > 0 else 'уже'} на {abs(aspect_diff):.1f} % при той же высоте "
                        f"(ширина / высота: референс {ref.aspect:.3f}, модель {model.aspect:.3f}).")

    # Полосы по высоте: (низ %, верх %, что, разница ширины, сдвиг левого края, сдвиг правого края).
    bands: list[list] = []
    for i in range(BANDS):
        y2 = round(rb.y2 - i * rb.h / BANDS)
        y1 = round(rb.y2 - (i + 1) * rb.h / BANDS)
        lo, hi = i * 100 // BANDS, (i + 1) * 100 // BANDS
        r_ext, m_ext = _extent(ref_mask, y1, y2), _extent(model_mask, y1, y2)
        if r_ext and not m_ext:
            bands.append([lo, hi, "у модели здесь ничего нет", 0.0, 0.0, 0.0])
            continue
        if m_ext and not r_ext:
            bands.append([lo, hi, "у модели лишнее, на референсе пусто", 0.0, 0.0, 0.0])
            continue
        if not r_ext:
            continue
        r_w, m_w = r_ext[1] - r_ext[0], m_ext[1] - m_ext[0]
        diff = (m_w - r_w) / r_w
        left = (r_ext[0] - m_ext[0]) / rb.w  # + значит край модели левее (шире наружу)
        right = (m_ext[1] - r_ext[1]) / rb.w
        if abs(diff) >= BAND_REPORT_DIFF or abs(left) >= EDGE_REPORT_SHIFT or abs(right) >= EDGE_REPORT_SHIFT:
            # Соседние полосы с почти тем же расхождением склеиваем в одну строку.
            prev = bands[-1] if bands else None
            if prev and prev[1] == lo and prev[2] == "width" and abs(prev[3] - diff) < 0.01 and abs(prev[4] - left) < 0.005 and abs(prev[5] - right) < 0.005:
                prev[1] = hi
            else:
                bands.append([lo, hi, "width", diff, left, right])
    lines = []
    for lo, hi, kind, diff, left, right in bands:
        if kind != "width":
            lines.append(f"  {lo}–{hi} %: {kind}")
            continue
        parts = [f"модель {'шире' if diff > 0 else 'уже'} на {abs(diff) * 100:.1f} %"]
        if abs(left) >= EDGE_REPORT_SHIFT:
            parts.append(f"левый край {'левее' if left > 0 else 'правее'} на {abs(left) * 100:.1f} % ширины")
        if abs(right) >= EDGE_REPORT_SHIFT:
            parts.append(f"правый край {'правее' if right > 0 else 'левее'} на {abs(right) * 100:.1f} % ширины")
        lines.append(f"  {lo}–{hi} %: " + ", ".join(parts))
    if lines:
        findings.append("По высоте (снизу вверх):")
        findings += lines

    area = float(ref_mask.sum())
    missing = _blobs(ref_mask & ~model_mask, rb, area)
    extra = _blobs(model_mask & ~ref_mask, rb, area)
    if missing:
        total = sum(a for a, _ in missing)
        findings.append(f"Не хватает (красное): {total * 100:.1f} % площади; " + "; ".join(f"{a * 100:.1f} % — {_where(b, rb)}" for a, b in missing[:BLOB_REPORT]))
    if extra:
        total = sum(a for a, _ in extra)
        findings.append(f"Лишнее (синее): {total * 100:.1f} % площади; " + "; ".join(f"{a * 100:.1f} % — {_where(b, rb)}" for a, b in extra[:BLOB_REPORT]))

    if ref.key_heights and model.key_heights:
        shifted = []
        for level in ref.key_heights:
            nearest = min(model.key_heights, key=lambda k: abs(k - level))
            if KEY_REPORT_DIFF <= abs(nearest - level) <= KEY_MATCH_TOL:
                shifted.append(f"{level * 100:.0f} % → у модели {nearest * 100:.0f} %")
        if shifted:
            findings.append("Ключевые высоты сдвинуты: " + "; ".join(shifted) + " (по силуэту; проверь глазами).")
    if model.parts != ref.parts:
        findings.append(f"Отдельных частей: референс {ref.parts}, модель {model.parts}.")
    if model.holes != ref.holes:
        findings.append(f"Сквозных отверстий: референс {ref.holes}, модель {model.holes}.")
    return Comparison(iou, s, ref, model, ref_mask, model_mask, model_rgb, findings, matrix)


def _contour(img: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], width: int) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, colour, width)


def panels(ref_img: Image.Image, c: Comparison, mode: str) -> list[tuple[str, Image.Image]]:
    """Картинки для показа: референс, модель, разница или наложение — всё в координатах референса."""
    ref_rgb = np.asarray(ref_img.convert("RGB")).astype(np.float32)
    width = max(2, round(max(ref_rgb.shape[:2]) / 400))
    b = c.ref.bbox
    pad = round(0.06 * max(b.w, b.h))
    crop = (max(0, b.x1 - pad), max(0, b.y1 - pad), min(ref_rgb.shape[1], b.x2 + pad), min(ref_rgb.shape[0], b.y2 + pad))

    def cut(a: np.ndarray) -> Image.Image:
        return Image.fromarray(a[crop[1] : crop[3], crop[0] : crop[2]].astype(np.uint8))

    result = []
    if mode in ("all", "side"):
        ref_view = ref_rgb.copy().astype(np.uint8)
        _contour(ref_view, c.ref_mask, REF_EDGE, width)
        model_view = c.model_rgb.copy()
        model_view[~c.model_mask] = (np.asarray(model_view[~c.model_mask], np.float32) * 0.3 + 128 * 0.7).astype(np.uint8)
        _contour(model_view, c.model_mask, MODEL_EDGE, width)
        result += [("референс", cut(ref_view)), ("модель", cut(model_view))]
    if mode in ("all", "diff"):
        faded = ref_rgb * 0.35 + 235 * 0.65
        diff = faded.copy()
        only_ref = c.ref_mask & ~c.model_mask
        only_model = c.model_mask & ~c.ref_mask
        diff[only_ref] = diff[only_ref] * 0.25 + np.array(MISSING) * 0.75
        diff[only_model] = diff[only_model] * 0.25 + np.array(EXTRA) * 0.75
        diff = diff.astype(np.uint8)
        _contour(diff, c.ref_mask, REF_EDGE, width)
        _contour(diff, c.model_mask, MODEL_EDGE, max(1, width // 2))
        result.append((f"разница · совпадение {c.iou * 100:.1f} %", cut(diff)))
    if mode == "overlay":
        blend = (ref_rgb * 0.5 + c.model_rgb.astype(np.float32) * 0.5).astype(np.uint8)
        _contour(blend, c.ref_mask, REF_EDGE, width)
        _contour(blend, c.model_mask, MODEL_EDGE, width)
        result.append((f"наложение 50/50 · совпадение {c.iou * 100:.1f} %", cut(blend)))
    return result


def compose_rows(rows: list[tuple[str, list[tuple[str, Image.Image]]]]) -> Image.Image:
    """Строки панелей одной высоты с подписями; у каждой строки свой заголовок."""
    row_h = PANEL_HEIGHT if len(rows) == 1 else max(360, PANEL_HEIGHT * 2 // (len(rows) + 1))
    scaled_rows = []
    for title, items in rows:
        scaled = [(label, img.resize((max(1, round(img.width * row_h / img.height)), row_h), Image.Resampling.LANCZOS)) for label, img in items]
        scaled_rows.append((title, scaled))
    width = max(sum(i.width for _, i in items) + PANEL_GAP * (len(items) + 1) for _, items in scaled_rows)
    title_h = LABEL_H if any(t for t, _ in rows) else 0
    height = sum(row_h + LABEL_H + title_h + PANEL_GAP for _ in scaled_rows) + PANEL_GAP
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    pen = ImageDraw.Draw(canvas)
    font, small = draw.font(28), draw.font(24)
    y = PANEL_GAP
    for title, items in scaled_rows:
        if title_h:
            pen.text((PANEL_GAP, y + title_h // 2), title, fill="white", font=font, anchor="lm")
            y += title_h
        x = PANEL_GAP
        for label, img in items:
            pen.text((x, y + LABEL_H // 2), label, fill=(230, 230, 230), font=small, anchor="lm")
            canvas.paste(img, (x, y + LABEL_H))
            x += img.width + PANEL_GAP
        y += LABEL_H + row_h + PANEL_GAP
    return canvas


LEGEND = "Зелёный контур — референс, пурпурный — модель; красное — не хватает у модели, синее — лишнее у модели."
