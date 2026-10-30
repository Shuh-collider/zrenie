"""Точные замеры: сетка координат, отметки Claude с привязкой к краю, контур и ось симметрии, палитра.

Все координаты на входе и выходе — пиксели оригинала. Числа считает код, а не модель на глаз.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import draw
from .layout import Box

GRID_COLOUR = (255, 0, 255)
GRID_TARGET_LINES = 10  # примерно столько подписанных линий по длинной стороне
GRID_MINOR_MIN_PX = 14  # тонкие линии рисуем, только если между ними на показе не меньше 14 px
RULER_MIN_GAP_PX = 70  # подписи на линейке — не чаще чем через 70 px показа
RULER_MINOR_MIN_PX = 8  # мелкие деления — если между ними на показе не меньше 8 px

SNAP_RADIUS = 12  # точку притягиваем к краю не дальше 12 px оригинала
MARK_COLOURS = [(255, 0, 255), (0, 200, 255), (255, 160, 0), (0, 220, 120), (255, 60, 60), (160, 100, 255)]
MARK_MARGIN_FRAC = 0.25  # вокруг отметок показываем ещё четверть их размаха

CONTOUR_DETAIL = 0.003  # точность упрощения контура: доля диагонали рамки объекта
CONTOUR_MAX_POINTS = 400
SYMMETRY_SEARCH_FRAC = 0.12  # ось ищем в ±12 % ширины от центра рамки
SYMMETRY_WORK_WIDTH = 512

PALETTE_SAMPLES = 60_000


def nice_step(span: float, target: int = GRID_TARGET_LINES) -> int:
    """Круглый шаг сетки (1, 2, 5 × 10ⁿ), чтобы по стороне было около target линий."""
    raw = max(1.0, span / target)
    power = 10 ** math.floor(math.log10(raw))
    for k in (1, 2, 5, 10):
        if k * power >= raw:
            return int(k * power)
    return int(10 * power)


def _label(pen: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, anchor: str, size: int) -> None:
    """Подпись на тёмной плашке — читается на любом фоне."""
    font = draw.font(size)
    left, top, right, bottom = pen.textbbox(xy, text, font=font, anchor=anchor)
    pen.rectangle([left - 2, top - 1, right + 2, bottom + 1], fill=(30, 30, 30, 200))
    pen.text(xy, text, fill=(255, 255, 255, 255), font=font, anchor=anchor)


def draw_pixel_grid(shown: Image.Image, region: Box, step: int) -> Image.Image:
    """Сетка в пикселях оригинала поверх показанной области. region — что показано (в оригинале)."""
    scale = region.w / shown.width  # пикселей оригинала на пиксель показа
    overlay = Image.new("RGBA", shown.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(overlay)
    minor = step / 5
    size = max(12, round(min(shown.size) / 60))
    if minor / scale >= GRID_MINOR_MIN_PX:
        for i in range(math.ceil(region.x1 / minor), math.floor(region.x2 / minor) + 1):
            x = (i * minor - region.x1) / scale
            pen.line([(x, 0), (x, shown.height)], fill=GRID_COLOUR + (45,), width=1)
        for i in range(math.ceil(region.y1 / minor), math.floor(region.y2 / minor) + 1):
            y = (i * minor - region.y1) / scale
            pen.line([(0, y), (shown.width, y)], fill=GRID_COLOUR + (45,), width=1)
    for value in range(math.ceil(region.x1 / step) * step, region.x2 + 1, step):
        x = (value - region.x1) / scale
        pen.line([(x, 0), (x, shown.height)], fill=GRID_COLOUR + (150,), width=1)
        _label(pen, (x, 2), str(value), "mt", size)
    for value in range(math.ceil(region.y1 / step) * step, region.y2 + 1, step):
        y = (value - region.y1) / scale
        pen.line([(0, y), (shown.width, y)], fill=GRID_COLOUR + (150,), width=1)
        _label(pen, (2, y), str(value), "lm", size)
    return Image.alpha_composite(shown.convert("RGBA"), overlay).convert("RGB")


def draw_percent_grid(shown: Image.Image, region: Box, obj: Box) -> Image.Image:
    """Сетка в процентах рамки объекта: 0 % — левый край и низ, 100 % — правый край и верх. obj — в оригинале."""
    scale = region.w / shown.width
    overlay = Image.new("RGBA", shown.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(overlay)
    size = max(12, round(min(shown.size) / 60))
    left, right = (obj.x1 - region.x1) / scale, (obj.x2 - region.x1) / scale
    top, bottom = (obj.y1 - region.y1) / scale, (obj.y2 - region.y1) / scale
    for pct in range(0, 101, 5):
        major = pct % 10 == 0
        alpha = 170 if pct in (0, 50, 100) else 110 if major else 40
        width = 2 if pct in (0, 50, 100) else 1
        x = left + (right - left) * pct / 100
        y = bottom - (bottom - top) * pct / 100
        pen.line([(x, top), (x, bottom)], fill=GRID_COLOUR + (alpha,), width=width)
        pen.line([(left, y), (right, y)], fill=GRID_COLOUR + (alpha,), width=width)
        if major:
            _label(pen, (x, max(0, top - 2)), f"{pct}", "mb" if top > size + 4 else "mt", size)
            _label(pen, (max(0, left - 4), y), f"{pct}", "rm" if left > 3 * size else "lm", size)
    return Image.alpha_composite(shown.convert("RGBA"), overlay).convert("RGB")


def draw_rulers(shown: Image.Image, region: Box) -> Image.Image:
    """Линейки по верхнему (x) и левому (y) краю с подписями в пикселях оригинала — координаты читаются с картинки.

    region — какая область оригинала показана. Размер картинки не меняется: деления лежат поверх краёв.
    """
    sx, sy = region.w / shown.width, region.h / shown.height  # пикселей оригинала на пиксель показа
    step = nice_step(RULER_MIN_GAP_PX * max(sx, sy), 1)
    k = 5 if step / 5 / max(sx, sy) >= RULER_MINOR_MIN_PX else 1  # делений на одну подпись
    minor = step / k
    size = max(12, round(min(shown.size) / 80))
    tick = max(6, size // 2)
    overlay = Image.new("RGBA", shown.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(overlay)
    colour = GRID_COLOUR + (255,)
    for i in range(math.ceil(region.x1 / minor), math.floor(region.x2 / minor) + 1):
        x = (i * minor - region.x1) / sx
        major = i % k == 0
        pen.line([(x, 0), (x, tick if major else tick // 2)], fill=colour, width=2 if major else 1)
        if major and 3 * size <= x <= shown.width - 2 * size:  # у левого края — подписи y
            _label(pen, (x, tick + 1), str(i // k * step), "mt", size)
    for i in range(math.ceil(region.y1 / minor), math.floor(region.y2 / minor) + 1):
        y = (i * minor - region.y1) / sy
        major = i % k == 0
        pen.line([(0, y), (tick if major else tick // 2, y)], fill=colour, width=2 if major else 1)
        if major and 2 * size <= y <= shown.height - size:
            _label(pen, (tick + 2, y), str(i // k * step), "lm", size)
    return Image.alpha_composite(shown.convert("RGBA"), overlay).convert("RGB")


@dataclass
class SnapResult:
    point: tuple[float, float]
    snapped: tuple[float, float] | None
    distance: float | None


def snap_points(points: list[tuple[float, float]], image: Image.Image, origin: tuple[int, int], mask: np.ndarray | None, mode: str) -> list[SnapResult]:
    """Притягивает точки к ближайшему краю: силуэта (mask) или любому резкому перепаду (edges)."""
    results = []
    gray = np.asarray(image.convert("L")) if mode == "edges" else None
    boundary = None
    if mode == "silhouette" and mask is not None:
        m = mask.astype(np.uint8)
        boundary = (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)
    for x, y in points:
        lx, ly = x - origin[0], y - origin[1]
        r = SNAP_RADIUS
        x1, y1 = max(0, int(lx) - r), max(0, int(ly) - r)
        x2, y2 = min(image.width, int(lx) + r + 1), min(image.height, int(ly) + r + 1)
        if x2 <= x1 or y2 <= y1:
            results.append(SnapResult((x, y), None, None))
            continue
        if mode == "silhouette" and boundary is not None:
            window = boundary[y1:y2, x1:x2]
        elif mode == "edges" and gray is not None:
            pad = 3
            patch = gray[max(0, y1 - pad) : y2 + pad, max(0, x1 - pad) : x2 + pad]
            edges = cv2.Canny(cv2.GaussianBlur(patch, (3, 3), 0), 40, 100, L2gradient=True) > 0
            window = edges[y1 - max(0, y1 - pad) : y1 - max(0, y1 - pad) + (y2 - y1), x1 - max(0, x1 - pad) : x1 - max(0, x1 - pad) + (x2 - x1)]
        else:
            results.append(SnapResult((x, y), None, None))
            continue
        ys, xs = np.nonzero(window)
        if len(xs) == 0:
            results.append(SnapResult((x, y), None, None))
            continue
        d = np.hypot(xs + x1 - lx, ys + y1 - ly)
        i = int(np.argmin(d))
        if d[i] > r:
            results.append(SnapResult((x, y), None, None))
            continue
        results.append(SnapResult((x, y), (float(xs[i] + x1 + origin[0]), float(ys[i] + y1 + origin[1])), float(d[i])))
    return results


LINE_SNAP_STEP = 0.25  # шаг поиска края вдоль линии, px
LINE_EDGE_MIN = 20.0  # слабее этого перепада (сумма по каналам на 1 px) — не край


def _sample(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Значения в дробных точках (билинейно). arr — H×W×C float32."""
    map_x = xs.astype(np.float32).reshape(-1, 1)
    map_y = ys.astype(np.float32).reshape(-1, 1)
    out = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(len(xs), -1)


def snap_line(line: tuple[float, float, float, float], image: Image.Image, origin: tuple[int, int], mask: np.ndarray | None, mode: str) -> tuple[tuple[float, float, float, float], list[float | None]]:
    """Двигает концы линии только вдоль неё — к краю, который линия пересекает рядом с концом.

    Так ширина и высота меряются точно и концы не перескакивают на соседнюю деталь сбоку.
    Возвращает новую линию и сдвиг каждого конца в px (None — края рядом нет).
    """
    x1, y1, x2, y2 = (line[0] - origin[0], line[1] - origin[1], line[2] - origin[0], line[3] - origin[1])
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 2:
        return line, [None, None]
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    if mode == "silhouette" and mask is not None:
        arr = mask.astype(np.float32)[..., None]
    else:
        arr = np.asarray(image.convert("RGB"), np.float32)
    r = SNAP_RADIUS
    s = np.arange(-r, r + LINE_SNAP_STEP / 2, LINE_SNAP_STEP)
    ends, shifts = [], []
    for ex, ey in ((x1, y1), (x2, y2)):
        best = None
        if mode == "silhouette" and mask is not None:
            v = _sample(arr, ex + s * ux, ey + s * uy)[:, 0] - 0.5
            crossings = np.nonzero(v[:-1] * v[1:] <= 0)[0]
            if len(crossings):
                i = crossings[np.argmin(np.abs(s[crossings]))]
                t = v[i] / (v[i] - v[i + 1]) if v[i] != v[i + 1] else 0.5
                best = s[i] + t * LINE_SNAP_STEP
        else:
            # Перепад на 1 px вдоль линии, усреднённый по трём параллельным линиям (меньше шума JPEG).
            mags = np.zeros(len(s))
            for off in (-1.0, 0.0, 1.0):
                px, py = ex - off * uy, ey + off * ux
                ahead = _sample(arr, px + (s + 0.5) * ux, py + (s + 0.5) * uy)
                behind = _sample(arr, px + (s - 0.5) * ux, py + (s - 0.5) * uy)
                mags += np.abs(ahead - behind).sum(axis=1)
            mags /= 3
            # Сильный край ближе к отметке Claude важнее сильного края на краю окна поиска.
            score = mags * np.exp(-((s / (r / 2)) ** 2))
            i = int(np.argmax(score))
            if mags[i] >= LINE_EDGE_MIN:
                best = s[i]
        if best is None:
            ends.append((ex, ey))
            shifts.append(None)
        else:
            ends.append((ex + best * ux, ey + best * uy))
            shifts.append(float(best))
    (a, b), (c, d) = ends
    return (a + origin[0], b + origin[1], c + origin[0], d + origin[1]), shifts


def marks_region(points: list[tuple[float, float]], lines: list[tuple[float, float, float, float]], bounds: Box) -> Box:
    """Область показа: все отметки и запас вокруг, но не меньше трети вида."""
    xs = [p[0] for p in points] + [v for line in lines for v in (line[0], line[2])]
    ys = [p[1] for p in points] + [v for line in lines for v in (line[1], line[3])]
    if not xs:
        return bounds
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    span = max(x2 - x1, y2 - y1, max(bounds.w, bounds.h) / 3)
    margin = MARK_MARGIN_FRAC * span
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = span / 2 + margin
    box = Box(int(cx - half), int(cy - half), int(math.ceil(cx + half)), int(math.ceil(cy + half)))
    return Box(max(bounds.x1, box.x1), max(bounds.y1, box.y1), min(bounds.x2, box.x2), min(bounds.y2, box.y2))


def draw_marks(shown: Image.Image, region: Box, points: list[tuple[float, float]], names: list[str], lines: list[tuple[float, float, float, float]]) -> Image.Image:
    scale = region.w / shown.width
    out = shown.convert("RGB")
    pen = ImageDraw.Draw(out)
    width = max(2, round(min(out.size) / 400))
    size = max(14, round(min(out.size) / 45))

    def to_shown(x: float, y: float) -> tuple[float, float]:
        return (x - region.x1) / scale, (y - region.y1) / scale

    for i, (x1, y1, x2, y2) in enumerate(lines):
        colour = MARK_COLOURS[i % len(MARK_COLOURS)]
        a, b = to_shown(x1, y1), to_shown(x2, y2)
        pen.line([a, b], fill=colour, width=width)
        for end in (a, b):
            pen.ellipse([end[0] - 2 * width, end[1] - 2 * width, end[0] + 2 * width, end[1] + 2 * width], outline=colour, width=width)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        _label(pen, mid, f"L{i + 1}", "mm", size)
    for i, ((x, y), name) in enumerate(zip(points, names)):
        colour = MARK_COLOURS[i % len(MARK_COLOURS)]
        cx, cy = to_shown(x, y)
        r = 3 * width
        pen.line([(cx - 3 * r, cy), (cx - r, cy)], fill=colour, width=width)
        pen.line([(cx + r, cy), (cx + 3 * r, cy)], fill=colour, width=width)
        pen.line([(cx, cy - 3 * r), (cx, cy - r)], fill=colour, width=width)
        pen.line([(cx, cy + r), (cx, cy + 3 * r)], fill=colour, width=width)
        pen.ellipse([cx - 1, cy - 1, cx + 1, cy + 1], fill=colour)
        _label(pen, (cx + 3 * r + 2, cy - 3 * r), name, "lb", size)
    return out


@dataclass
class Symmetry:
    axis_x: float  # в координатах маски (вида)
    score: float  # доля совпадения силуэта с зеркальным (IoU), 1 — идеально симметричен


def find_symmetry(mask: np.ndarray, bbox: Box) -> Symmetry:
    """Вертикальная ось, относительно которой силуэт больше всего похож на своё отражение."""
    crop = mask[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
    k = min(1.0, SYMMETRY_WORK_WIDTH / max(1, crop.shape[1]))
    if k < 1:
        crop = cv2.resize(crop.astype(np.uint8), (max(1, round(crop.shape[1] * k)), max(1, round(crop.shape[0] * k))), interpolation=cv2.INTER_NEAREST).astype(bool)
    w = crop.shape[1]
    pad = w  # с запасом, чтобы сдвиг не обрезал маску
    padded = np.zeros((crop.shape[0], w + 2 * pad), bool)
    padded[:, pad : pad + w] = crop
    flipped = padded[:, ::-1]
    best_s, best_iou = 0, -1.0
    search = max(1, round(SYMMETRY_SEARCH_FRAC * w))
    for s in range(-2 * search, 2 * search + 1):  # ось на (w - 1 + s) / 2 в координатах crop
        shifted = np.roll(flipped, s, axis=1)
        inter = np.logical_and(padded, shifted).sum()
        union = np.logical_or(padded, shifted).sum()
        iou = inter / union if union else 0.0
        if iou > best_iou:
            best_s, best_iou = s, iou
    axis_crop = (w - 1 + best_s) / 2 + 0.5  # +0.5: ось между пикселями, в координатах краёв
    return Symmetry(axis_x=bbox.x1 + axis_crop / k, score=float(best_iou))


@dataclass
class Contour:
    kind: str  # "outer" | "hole"
    points: np.ndarray  # N×2, пиксели вида (x, y)


def find_contours(mask: np.ndarray, bbox: Box, detail: float) -> list[Contour]:
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    epsilon = max(0.5, detail * math.hypot(bbox.w, bbox.h))
    total_area = mask.sum()
    result = []
    for contour, (_, _, _, parent) in zip(contours, hierarchy[0]):
        if cv2.contourArea(contour) < 0.001 * total_area:
            continue
        # +0.5: координаты центров пикселей, как у рамки и оси (края пикселей — целые числа).
        simple = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float64) + 0.5
        result.append(Contour("outer" if parent == -1 else "hole", simple))
    result.sort(key=lambda c: (c.kind != "outer", -cv2.contourArea(c.points.astype(np.float32))))
    return result


def half_profile(outer: np.ndarray, axis_x: float) -> np.ndarray:
    """Правая половина внешнего контура от верха до низа — профиль для тела вращения (Screw / Lathe).

    Начинается и кончается точно на оси (x = axis_x), чтобы тело вращения получилось закрытым.
    """
    n = len(outer)
    keep = outer[:, 0] >= axis_x
    if not keep.any():
        return np.empty((0, 2))
    if keep.all():
        return outer[np.argsort(outer[:, 1])]
    # Сдвигаем контур так, чтобы он начинался с точки слева от оси, — тогда правая часть идёт подряд.
    shift = int(np.argmin(keep))
    pts, right = np.roll(outer, -shift, axis=0), np.roll(keep, -shift)
    best, i = (0, 0), 0
    while i < n:
        if right[i]:
            j = i
            while j < n and right[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    start, end = best

    def crossing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        t = (axis_x - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
        return np.array([axis_x, a[1] + t * (b[1] - a[1])])

    profile = np.vstack([crossing(pts[start - 1], pts[start]), pts[start:end], crossing(pts[end - 1], pts[end % n])])
    if profile[0, 1] > profile[-1, 1]:
        profile = profile[::-1]
    return profile


def to_scene(points: np.ndarray, axis_x: float, bottom_y: float, units_per_px: float) -> np.ndarray:
    """Пиксели → координаты сцены: начало на оси у низа объекта, x вправо, z вверх."""
    return np.stack([(points[:, 0] - axis_x) * units_per_px, (bottom_y - points[:, 1]) * units_per_px], axis=1)


def draw_contours(shown: Image.Image, scale: float, contours: list[Contour], axis_x: float | None) -> Image.Image:
    out = shown.convert("RGB")
    pen = ImageDraw.Draw(out)
    width = max(2, round(min(out.size) / 400))
    for c in contours:
        colour = (255, 0, 255) if c.kind == "outer" else (0, 200, 255)
        pts = [(x / scale, y / scale) for x, y in c.points]
        pen.line(pts + [pts[0]], fill=colour, width=width)
        for x, y in pts:
            pen.ellipse([x - width, y - width, x + width, y + width], fill=colour)
    if axis_x is not None:
        x = axis_x / scale
        for y in range(0, out.height, 18):
            pen.line([(x, y), (x, y + 9)], fill=(0, 220, 120), width=width)
    return out


@dataclass
class Swatch:
    rgb: tuple[int, int, int]
    share: float

    @property
    def hex(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)

    @property
    def srgb01(self) -> tuple[float, float, float]:
        return tuple(round(c / 255, 3) for c in self.rgb)

    @property
    def linear01(self) -> tuple[float, float, float]:
        def lin(c: float) -> float:
            c /= 255
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        return tuple(round(lin(c), 3) for c in self.rgb)


def palette(image: Image.Image, mask: np.ndarray | None, colours: int) -> list[Swatch]:
    """Основные цвета (k-means в Lab) по пикселям объекта, от самого частого."""
    rgb = np.asarray(image.convert("RGB"))
    pixels = rgb[mask] if mask is not None and mask.any() else rgb.reshape(-1, 3)
    if len(pixels) > PALETTE_SAMPLES:
        pixels = pixels[np.random.default_rng(0).choice(len(pixels), PALETTE_SAMPLES, replace=False)]
    # Группируем в Lab (близость там похожа на то, как различает глаз), а цвет группы — среднее её пикселей в RGB.
    lab = cv2.cvtColor((pixels.astype(np.float32) / 255).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    k = max(1, min(colours, len(lab)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    cv2.setRNGSeed(0)
    _, labels, _ = cv2.kmeans(lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    counts = np.bincount(labels, minlength=k)
    order = [i for i in np.argsort(-counts) if counts[i] > 0]
    return [
        Swatch(tuple(int(round(v)) for v in pixels[labels == i].mean(axis=0)), float(counts[i] / counts.sum()))
        for i in order
    ]


def draw_palette(thumb: Image.Image, swatches: list[Swatch]) -> Image.Image:
    """Слева — объект, справа — полосы цветов с HEX и долей."""
    thumb = thumb.convert("RGB")
    thumb.thumbnail((700, 700))
    row = max(48, thumb.height // max(1, len(swatches)))
    width = thumb.width + 380
    height = max(thumb.height, row * len(swatches))
    out = Image.new("RGB", (width, height), (60, 60, 60))
    out.paste(thumb, (0, 0))
    pen = ImageDraw.Draw(out)
    font = draw.font(22)
    for i, s in enumerate(swatches):
        y = i * row
        pen.rectangle([thumb.width + 10, y + 4, thumb.width + 140, y + row - 4], fill=s.rgb)
        pen.text((thumb.width + 150, y + row / 2), f"{i + 1}. {s.hex}  {s.share * 100:.0f} %", fill="white", font=font, anchor="lm")
    return out
