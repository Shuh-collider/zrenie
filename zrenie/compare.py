"""Сравнение рендера модели с референсом: выравнивание силуэтов, совпадение (IoU), где и насколько расходятся."""

import math
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
# Модель шире кадра референса не обрезаем: холст расширяется, но не больше чем на размер вида с каждой стороны,
# чтобы пол или фон на рендере не раздули его до гигантского.
CANVAS_MAX_GROWTH = 1.0
CANVAS_MARGIN_PX = 2  # запас на сглаживание края при масштабировании
# Сначала выравниваем по высоте и низу рамки. Если рамки не совпадают (у модели лишняя антенна или нет верхней детали),
# масштаб сбивается у всей модели и расхождения видны везде — тогда ищем масштаб и сдвиг, при которых силуэты совпадают
# больше всего. Масштаб — относительно выравнивания по рамке; поиск — на уменьшенных масках, сначала грубо, потом точнее.
ALIGN_SCALES = np.geomspace(0.6, 1.6, 41)
ALIGN_FINE = np.geomspace(0.97, 1.03, 9)
ALIGN_COARSE_PX, ALIGN_FINE_PX = 256, 512
ALIGN_MAX_SIZE = 3.0  # модель больше вида втрое и сильнее (пол, фон на рендере) — по совпадению не выравниваем
ALIGN_MIN_GAIN = 0.05  # по совпадению — только если оно лучше, чем по рамке, хотя бы на 5 пунктов
ALIGNED_BY_BOX = "по высоте и низу"
ALIGNED_BY_OVERLAP = "по наибольшему совпадению силуэтов"

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
    """Всё — в координатах холста: вид референса с запасом по краям, чтобы модель шире кадра вошла целиком."""

    iou: float
    scale: float  # во сколько раз увеличен рендер, чтобы совпасть по высоте
    ref: Shape
    model: Shape  # силуэт модели после выравнивания
    ref_mask: np.ndarray
    model_mask: np.ndarray
    ref_rgb: np.ndarray  # референс на холсте
    model_rgb: np.ndarray  # рендер на холсте (для наложения)
    findings: list[str]
    matrix: np.ndarray  # 2×3: пиксели рендера → пиксели холста (для деталей модели)
    offset: tuple[int, int]  # где на холсте левый верхний угол вида референса
    aligned_by: str  # ALIGNED_BY_BOX или ALIGNED_BY_OVERLAP

    @property
    def ref_box(self) -> Box:
        """Рамка объекта референса в координатах вида."""
        b, (ox, oy) = self.ref.bbox, self.offset
        return Box(b.x1 - ox, b.y1 - oy, b.x2 - ox, b.y2 - oy)

    def to_canvas(self, mask: np.ndarray) -> np.ndarray:
        """Маска в координатах вида референса → на холст."""
        (ox, oy), (h, w) = self.offset, mask.shape
        canvas_h, canvas_w = self.ref_mask.shape
        return np.pad(mask, ((oy, canvas_h - h - oy), (ox, canvas_w - w - ox)))

    def shown_area(self) -> tuple[int, int, int, int]:
        """Что показывать на картинках: референс и модель целиком и запас 6 % вокруг."""
        r, m = self.ref.bbox, self.model.bbox
        x1, y1, x2, y2 = min(r.x1, m.x1), min(r.y1, m.y1), max(r.x2, m.x2), max(r.y2, m.y2)
        pad = round(0.06 * max(x2 - x1, y2 - y1))
        h, w = self.ref_mask.shape
        return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)


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


def _best_overlap(ref_mask: np.ndarray, model_mask: np.ndarray, model_box: Box, s0: float) -> tuple[float, float, float, float]:
    """Масштаб и сдвиг рендера, при которых силуэты совпадают больше всего: (s, tx, ty, совпадение — приблизительно).

    Перебираем масштаб, а лучший сдвиг для каждого масштаба сразу даёт свёртка масок (matchTemplate)."""
    crop = model_mask[model_box.y1 : model_box.y2, model_box.x1 : model_box.x2].astype(np.float32)
    ref = ref_mask.astype(np.float32)

    def search(factors: np.ndarray, work_px: int) -> tuple[float, float, float, float]:
        k = min(1.0, work_px / max(ref.shape))
        small = cv2.resize(ref, (max(1, round(ref.shape[1] * k)), max(1, round(ref.shape[0] * k))), interpolation=cv2.INTER_AREA)
        area = float(small.sum())
        best = (-1.0, 1.0, 0.0, 0.0)
        for f in factors:
            mw, mh = max(1, round(model_box.w * s0 * f * k)), max(1, round(model_box.h * s0 * f * k))
            if mw > ALIGN_MAX_SIZE * small.shape[1] or mh > ALIGN_MAX_SIZE * small.shape[0]:
                continue
            m = cv2.resize(crop, (mw, mh), interpolation=cv2.INTER_AREA)
            padded = cv2.copyMakeBorder(small, mh, mh, mw, mw, cv2.BORDER_CONSTANT, value=0)
            inter = cv2.matchTemplate(padded, m, cv2.TM_CCORR)  # площадь пересечения при каждом сдвиге
            iou = inter / np.maximum(area + float(m.sum()) - inter, 1e-6)
            _, value, _, (px, py) = cv2.minMaxLoc(iou)
            if value > best[0]:  # левый верхний угол рамки модели — в пикселях вида референса
                best = (value, float(f), (px - mw) / k, (py - mh) / k)
        return best

    _, f, _, _ = search(ALIGN_SCALES, ALIGN_COARSE_PX)
    iou, f, x0, y0 = search(f * ALIGN_FINE, ALIGN_FINE_PX)
    s = s0 * f
    return s, x0 - model_box.x1 * s, y0 - model_box.y1 * s, iou


def compare(ref_img: Image.Image, ref_matte: Matte, render_img: Image.Image, render_matte: Matte) -> Comparison | None:
    ref0 = measure(ref_matte)
    model0 = measure(render_matte)
    if ref0 is None or model0 is None:
        return None
    # Выравниваем по высоте и низу: рендер масштабируется так, чтобы высота силуэта совпала, низ и центр рамки — тоже.
    s = ref0.bbox.h / model0.bbox.h
    m_cx, r_cx = (model0.bbox.x1 + model0.bbox.x2) / 2, (ref0.bbox.x1 + ref0.bbox.x2) / 2
    by_box = _compare_aligned(ref_img, ref_matte, render_img, render_matte, model0, s, r_cx - m_cx * s, ref0.bbox.y2 - model0.bbox.y2 * s)
    if by_box.iou >= 1 - ALIGN_MIN_GAIN:  # лучше на 5 пунктов уже не станет
        return by_box
    s2, tx2, ty2, estimate = _best_overlap(ref_matte.mask, render_matte.mask, model0.bbox, s)
    if estimate < by_box.iou + ALIGN_MIN_GAIN:
        return by_box
    by_overlap = _compare_aligned(ref_img, ref_matte, render_img, render_matte, model0, s2, tx2, ty2, by_box.iou)
    return by_overlap if by_overlap.iou >= by_box.iou + ALIGN_MIN_GAIN else by_box


def _compare_aligned(
    ref_img: Image.Image,
    ref_matte: Matte,
    render_img: Image.Image,
    render_matte: Matte,
    model0: Shape,
    s: float,
    tx: float,
    ty: float,
    box_iou: float | None = None,
) -> Comparison:
    """Сравнение при заданном выравнивании: пиксели вида референса = s × пиксели рендера + (tx, ty).

    box_iou — совпадение при выравнивании по рамке, если это выравнивание по совпадению силуэтов (для отчёта)."""
    height, width = ref_matte.mask.shape
    # Холст — вид референса с запасом по краям, чтобы модель, которая шире кадра, вошла целиком:
    # обрезанная модель занижает расхождения. Запас слева, сверху, справа, снизу — и его потолок.
    need = (
        CANVAS_MARGIN_PX - (model0.bbox.x1 * s + tx),
        CANVAS_MARGIN_PX - (model0.bbox.y1 * s + ty),
        model0.bbox.x2 * s + tx + CANVAS_MARGIN_PX - width,
        model0.bbox.y2 * s + ty + CANVAS_MARGIN_PX - height,
    )
    grow_x, grow_y = round(CANVAS_MAX_GROWTH * width), round(CANVAS_MAX_GROWTH * height)
    limits = (grow_x, grow_y, grow_x, grow_y)
    pad_left, pad_top, pad_right, pad_bottom = (min(limit, max(0, math.ceil(n))) for n, limit in zip(need, limits))
    clipped = any(n > limit for n, limit in zip(need, limits))
    pads = ((pad_top, pad_bottom), (pad_left, pad_right))
    ref_alpha = np.pad(ref_matte.alpha, pads)
    ref_rgb = np.pad(np.asarray(ref_img.convert("RGB")), (*pads, (0, 0)), mode="edge")
    ref = measure(Matte(ref_alpha, ref_matte.method))
    ref_mask = ref_alpha >= 0.5
    height, width = ref_mask.shape

    matrix = np.float32([[s, 0, tx + pad_left], [0, s, ty + pad_top]])
    warped_alpha = cv2.warpAffine(render_matte.alpha, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)
    model_mask = warped_alpha >= 0.5
    model_rgb = cv2.warpAffine(np.asarray(render_img.convert("RGB")), matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
    model = measure(Matte(warped_alpha, render_matte.method))
    union = np.logical_or(ref_mask, model_mask).sum()
    iou = float(np.logical_and(ref_mask, model_mask).sum() / union) if union else 0.0

    findings: list[str] = []
    if clipped:
        findings.append("Модель намного больше кадра референса — часть за краем не учтена. Проверь, нет ли на рендере лишнего: пола, фона, других объектов.")
    rb = ref.bbox
    if box_iou is None:
        # Пропорции — по рендеру до выравнивания: масштаб их не меняет, а холст не обрезает.
        aspect_diff = (model0.aspect - ref.aspect) / ref.aspect * 100
        if abs(aspect_diff) >= BAND_REPORT_DIFF * 100:
            findings.append(f"В целом модель {'шире' if aspect_diff > 0 else 'уже'} на {abs(aspect_diff):.1f} % при той же высоте "
                            f"(ширина / высота: референс {ref.aspect:.3f}, модель {model0.aspect:.3f}).")
    else:
        dw, dh = (model.bbox.w - rb.w) / rb.w * 100, (model.bbox.h - rb.h) / rb.h * 100
        findings.append(
            f"Рамки силуэтов не совпадают: у модели ширина {dw:+.1f} %, высота {dh:+.1f} % от референса. Поэтому модель выровнена "
            f"по наибольшему совпадению силуэтов (по высоте и низу было бы {box_iou * 100:.1f} %) — смотри, что лишнее и чего не хватает."
        )

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
        # Ключевые высоты модели — в долях рамки референса: при выравнивании по совпадению рамки разной высоты.
        mb = model.bbox
        model_levels = [(rb.y2 - (mb.y2 - k * mb.h)) / rb.h for k in model.key_heights]
        shifted = []
        for level in ref.key_heights:
            nearest = min(model_levels, key=lambda k: abs(k - level))
            if KEY_REPORT_DIFF <= abs(nearest - level) <= KEY_MATCH_TOL:
                shifted.append(f"{level * 100:.0f} % → у модели {nearest * 100:.0f} %")
        if shifted:
            findings.append("Ключевые высоты сдвинуты: " + "; ".join(shifted) + " (по силуэту; проверь глазами).")
    if model.parts != ref.parts:
        findings.append(f"Отдельных частей: референс {ref.parts}, модель {model.parts}.")
    if model.holes != ref.holes:
        findings.append(f"Сквозных отверстий: референс {ref.holes}, модель {model.holes}.")
    aligned_by = ALIGNED_BY_BOX if box_iou is None else ALIGNED_BY_OVERLAP
    return Comparison(iou, s, ref, model, ref_mask, model_mask, ref_rgb, model_rgb, findings, matrix, (pad_left, pad_top), aligned_by)


def _contour(img: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], width: int) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, colour, width)


def panels(c: Comparison, mode: str) -> list[tuple[str, Image.Image]]:
    """Картинки для показа: референс, модель, разница или наложение — всё на холсте сравнения."""
    ref_rgb = c.ref_rgb.astype(np.float32)
    width = max(2, round(max(ref_rgb.shape[:2]) / 400))
    crop = c.shown_area()

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
