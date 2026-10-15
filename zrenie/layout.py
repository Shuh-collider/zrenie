"""Поиск отдельных кадров на листе: коллаж с промежутками, коллаж «встык» (швы), виды на общем фоне.

Алгоритм — рекурсивный XY-разрез:
1. Промежуток — линия (строка или столбец) без краёв и почти одного цвета: белая полоса между кадрами,
   пустой фон между видами, поля по краям. Режем по промежуткам, поля обрезаем.
2. Если промежутков нет — ищем шов: прямую линию через весь кусок, на которой картинка резко меняется,
   а рядом — нет. Так склеены кадры коллажа без промежутков.
3. Повторяем для каждого куска. Мелкие куски (подписи, тени) отбрасываем.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

# Анализ на уменьшенной копии — быстро и почти без потери точности.
ANALYSIS_MAX_EDGE = 1200

# Промежуток: соседние пиксели вдоль линии почти не отличаются и вся линия почти одного цвета
# (разброс между 2-м и 98-м перцентилем не больше CONST_RANGE по каждому каналу).
EDGE_STEP = 12
EDGE_FREE_FRAC = 0.01
CONST_RANGE = 24
CONST_PERCENTILES = (2, 98)

# Промежуток внутри куска принимаем, только если он узкий (полоса между кадрами) или цвета фона
# (цвет по краям куска). Иначе это может быть ровная заливка самого объекта во всю ширину.
THIN_GAP_FRAC = 0.03
BACKGROUND_TOL = 24
# Внутренний промежуток тоньше этого — не граница кадров, а щель в объекте (полоска стекла, зазор).
MIN_GAP_FRAC = 0.003
MIN_GAP_PX = 3

# Куски короче этого (доля стороны картинки) — не кадры, а шум у края объекта.
MIN_SEGMENT_FRAC = 0.04
MIN_SEGMENT_PX = 16

# Шов: ступенька яркости между средними по 3 px с каждой стороны.
SEAM_K = 3
SEAM_STEP = 2.0
SEAM_STRONG_STEP = 8.0
SEAM_MIN_FRAC = 0.3
SEAM_MIN_RATIO = 4.0  # во сколько раз чаще слабая ступенька, чем у соседних линий
SEAM_STRONG_MIN_FRAC = 0.4
SEAM_STRONG_MIN_RATIO = 3.0  # то же для сильной ступеньки
# Шов идёт через весь кусок: делим линию на 10 частей, в 9 из них ступенька должна быть хотя бы у 30 % точек.
SEAM_DECILE_COVER = 0.3
SEAM_MIN_DECILES = 9
# Внутри кадра (не первый разрез листа) горизонт или край стола выглядит как шов. Поэтому там шов
# принимаем, только если сильная ступенька есть на каждой десятой части линии хотя бы у 60 % точек.
INNER_SEAM_DECILE_COVER = 0.6
# Шов ищем только от края до края кадра: края куска — край листа или прошлый шов,
# а пустых полей срезано не больше 5 %. Так не режем один объект по его прямой линии.
HARD_TRIM_FRAC = 0.05
SEAM_NEIGHBOUR_NEAR = 4
SEAM_NEIGHBOUR_FAR = 15
# Слабый шов принимаем только там, где генераторы обычно делят лист: 1/2, 1/3, 2/3, 1/4, 3/4.
GRID_FRACTIONS = (1 / 2, 1 / 3, 2 / 3, 1 / 4, 3 / 4)
GRID_TOL = 0.02

# Точная граница шва ищется на оригинале в ±4 px (в масштабе анализа) от найденной; по 2 px с каждой
# стороны шва отрезаем — там пиксели смешаны из обоих кадров.
SEAM_SNAP_RADIUS = 4
SEAM_INSET_PX = 2

MAX_DEPTH = 5
MAX_VIEWS = 16
MIN_AREA_FRAC = 0.05  # кусок меньше 5 % самого большого — подпись или тень
# Узкая полоса (подпись, строка текста), если она не главный кадр.
STRIP_ASPECT = 6.0
STRIP_MAX_AREA_FRAC = 0.2
PAD_FRAC = 0.03


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int  # не включая
    y2: int  # не включая

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Layout:
    views: list[Box]  # кадры в порядке чтения, с небольшим запасом по краям
    content: Box  # где вообще есть изображение, без пустых полей
    method: str
    dropped: int  # сколько мелких кусков отброшено


def _line_view(a: np.ndarray, axis: str) -> np.ndarray:
    """Возвращает массив, где b[:, i] — i-я линия: столбец при axis='x', строка при axis='y'."""
    return a if axis == "x" else a.transpose(1, 0, 2)


def _gap_mask(region: np.ndarray, axis: str) -> np.ndarray:
    b = _line_view(region, axis)
    if b.shape[0] < 2:
        return np.zeros(b.shape[1], dtype=bool)
    steps = np.abs(np.diff(b, axis=0)).max(axis=2)
    edge_free = (steps > EDGE_STEP).mean(axis=0) < EDGE_FREE_FRAC
    lo, hi = np.percentile(b, CONST_PERCENTILES, axis=0)
    constant = (hi - lo).max(axis=1) <= CONST_RANGE
    return edge_free & constant


def _gap_runs(gap: np.ndarray) -> list[tuple[int, int]]:
    runs, i, n = [], 0, len(gap)
    while i < n:
        if gap[i]:
            j = i
            while j < n and gap[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _keep_background_gaps(gap: np.ndarray, region: np.ndarray, axis: str, fallback: np.ndarray) -> np.ndarray:
    """Убирает внутренние «промежутки», которые толстые и не цвета фона, — это заливка объекта.

    Цвет фона — цвет пустых полей у краёв куска; если полей нет — цвет краёв всего листа.
    """
    b = _line_view(region, axis)
    n = len(gap)
    runs = _gap_runs(gap)
    edge_runs = [(i, j) for i, j in runs if i == 0 or j == n]
    if edge_runs:
        background = np.median(np.concatenate([b[:, i:j].reshape(-1, 3) for i, j in edge_runs]), axis=0)
    else:
        background = fallback
    out = gap.copy()
    min_gap = max(MIN_GAP_PX, round(MIN_GAP_FRAC * n))
    for i, j in runs:
        if (i, j) in edge_runs:
            continue
        if j - i < min_gap:
            out[i:j] = False
            continue
        if j - i <= THIN_GAP_FRAC * n:
            continue
        colour = np.median(b[:, i:j].reshape(-1, 3), axis=0)
        if np.abs(colour - background).max() > BACKGROUND_TOL:
            out[i:j] = False
    return out


def _segments(gap: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Отрезки без промежутков [начало, конец), короткие отбрасываются."""
    segments, start = [], None
    for i, is_gap in enumerate(gap):
        if not is_gap and start is None:
            start = i
        elif is_gap and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(gap)))
    return [(s, e) for s, e in segments if e - s >= min_len]


def _neighbour_ratio(values: np.ndarray) -> np.ndarray:
    """Во сколько раз значение больше медианы соседей (без ближайших линий)."""
    ratio = np.zeros_like(values)
    n = len(values)
    for i in range(n):
        lo, hi = max(0, i - SEAM_NEIGHBOUR_FAR), min(n, i + SEAM_NEIGHBOUR_FAR + 1)
        near_lo, near_hi = max(lo, i - SEAM_NEIGHBOUR_NEAR), min(hi, i + SEAM_NEIGHBOUR_NEAR + 1)
        neighbours = np.concatenate([values[lo:near_lo], values[near_hi:hi]])
        ratio[i] = values[i] / (float(np.median(neighbours)) + 0.05)
    return ratio


def _near_grid(rel: np.ndarray) -> np.ndarray:
    return np.min(np.abs(rel[:, None] - np.array(GRID_FRACTIONS)[None]), axis=1) <= GRID_TOL


def _seams(region: np.ndarray, axis: str, min_len: int, offset: int, full: int, anywhere_ok: bool) -> tuple[list[int], float]:
    """Позиции швов (граница перед этой линией, внутри куска) и сила лучшего шва.

    offset — где кусок начинается на листе; full — длина стороны листа.
    Шов принимаем только на «круглых» долях листа (1/2, 1/3, …): горизонт или край стола на рендере в любом
    другом месте выглядит так же, как шов. anywhere_ok — первый разрез всего листа: там хватает слабого шва;
    внутри кадра — только очень сильного без перерывов.
    """
    b = _line_view(region, axis)
    n = b.shape[1]
    k = SEAM_K
    if n < 2 * min_len or n <= 2 * k + 2 * SEAM_NEIGHBOUR_FAR:
        return [], 0.0

    csum = np.cumsum(np.pad(b, ((0, 0), (1, 0), (0, 0))), axis=1)
    xs = np.arange(k, n - k)
    left = (csum[:, xs + 1] - csum[:, xs + 1 - k]) / k
    right = (csum[:, xs + 1 + k] - csum[:, xs + 1]) / k
    step = np.abs(left - right).max(axis=2)
    weak = step > SEAM_STEP
    strong_mask = step > SEAM_STRONG_STEP
    frac = weak.mean(axis=0)
    strong = strong_mask.mean(axis=0)

    def deciles(mask: np.ndarray) -> np.ndarray:
        return sum((part.mean(axis=0) >= SEAM_DECILE_COVER).astype(int) for part in np.array_split(mask, 10, axis=0))

    deciles_weak, deciles_strong = deciles(weak), deciles(strong_mask)
    ratio_weak = _neighbour_ratio(frac)
    ratio_strong = _neighbour_ratio(strong)

    positions = xs + 1
    absolute = positions + offset
    near_grid = _near_grid(absolute / full)
    # Каждое свидетельство проверяем на всей длине шва отдельно: шум даёт слабые ступеньки везде,
    # а край объекта в одном кадре даёт сильную ступеньку только на части линии.
    if anywhere_ok:
        weak_seam = (ratio_weak >= SEAM_MIN_RATIO) & (frac >= SEAM_MIN_FRAC) & (deciles_weak >= SEAM_MIN_DECILES)
        strong_seam = (
            (ratio_strong >= SEAM_STRONG_MIN_RATIO) & (strong >= SEAM_STRONG_MIN_FRAC) & (deciles_strong >= SEAM_MIN_DECILES)
        )
        found = near_grid & (weak_seam | strong_seam)
    else:
        min_cover = np.min(np.stack([part.mean(axis=0) for part in np.array_split(strong_mask, 10, axis=0)]), axis=0)
        found = near_grid & (ratio_strong >= SEAM_STRONG_MIN_RATIO) & (min_cover >= INNER_SEAM_DECILE_COVER)
    ok = found & (positions >= min_len) & (positions <= n - min_len)
    score = np.maximum(ratio_weak, ratio_strong)

    accepted: list[int] = []
    best = 0.0
    for i in np.argsort(-score):
        if not ok[i]:
            continue
        if all(abs(positions[i] - p) > 2 * SEAM_NEIGHBOUR_NEAR for p in accepted):
            accepted.append(int(positions[i]))
            best = max(best, float(score[i]))
    return sorted(accepted), best


@dataclass(frozen=True)
class _Hard:
    """Какие края куска «жёсткие»: край листа или шов, а не пустой фон."""

    left: bool = True
    top: bool = True
    right: bool = True
    bottom: bool = True


class _Splitter:
    def __init__(self, raw: np.ndarray, smooth: np.ndarray):
        self.raw, self.smooth = raw, smooth
        self.height, self.width = raw.shape[:2]
        self.min_x = max(MIN_SEGMENT_PX, int(MIN_SEGMENT_FRAC * self.width))
        self.min_y = max(MIN_SEGMENT_PX, int(MIN_SEGMENT_FRAC * self.height))
        self.used_gaps = False
        self.used_seams = False
        # Швы, по которым резали: (ось, позиция границы, начало и конец линии) в масштабе анализа.
        self.seam_lines: list[tuple[str, int, int, int]] = []
        border = np.concatenate([smooth[:2].reshape(-1, 3), smooth[-2:].reshape(-1, 3), smooth[:, :2].reshape(-1, 3), smooth[:, -2:].reshape(-1, 3)])
        self.background = np.median(border, axis=0)

    def split(self, box: Box, hard: _Hard = _Hard(), depth: int = 0) -> list[Box]:
        if depth > MAX_DEPTH or box.w < self.min_x or box.h < self.min_y:
            return [box]
        smooth = self.smooth[box.y1 : box.y2, box.x1 : box.x2]
        segs_x = _segments(_keep_background_gaps(_gap_mask(smooth, "x"), smooth, "x", self.background), self.min_x)
        segs_y = _segments(_keep_background_gaps(_gap_mask(smooth, "y"), smooth, "y", self.background), self.min_y)
        if not segs_x or not segs_y:
            return []  # пустой кусок

        if len(segs_y) > 1 or len(segs_x) > 1:
            self.used_gaps = True
            if len(segs_y) >= len(segs_x):
                x1, x2 = box.x1 + segs_x[0][0], box.x1 + segs_x[-1][1]
                parts = [
                    (Box(x1, box.y1 + s, x2, box.y1 + e), _Hard(left=False, top=False, right=False, bottom=False))
                    for s, e in segs_y
                ]
            else:
                y1, y2 = box.y1 + segs_y[0][0], box.y1 + segs_y[-1][1]
                parts = [
                    (Box(box.x1 + s, y1, box.x1 + e, y2), _Hard(left=False, top=False, right=False, bottom=False))
                    for s, e in segs_x
                ]
            return [leaf for part, part_hard in parts for leaf in self.split(part, part_hard, depth + 1)]

        trimmed = Box(box.x1 + segs_x[0][0], box.y1 + segs_y[0][0], box.x1 + segs_x[0][1], box.y1 + segs_y[0][1])
        hard = _Hard(
            left=hard.left and trimmed.x1 - box.x1 <= HARD_TRIM_FRAC * box.w,
            top=hard.top and trimmed.y1 - box.y1 <= HARD_TRIM_FRAC * box.h,
            right=hard.right and box.x2 - trimmed.x2 <= HARD_TRIM_FRAC * box.w,
            bottom=hard.bottom and box.y2 - trimmed.y2 <= HARD_TRIM_FRAC * box.h,
        )
        raw = self.raw[trimmed.y1 : trimmed.y2, trimmed.x1 : trimmed.x2]
        seams_x, score_x = [], 0.0
        seams_y, score_y = [], 0.0
        if hard.top and hard.bottom:  # вертикальный шов идёт сверху донизу кадра
            seams_x, score_x = _seams(raw, "x", self.min_x, trimmed.x1, self.width, depth == 0)
        if hard.left and hard.right:
            seams_y, score_y = _seams(raw, "y", self.min_y, trimmed.y1, self.height, depth == 0)
        if not seams_x and not seams_y:
            return [trimmed]

        self.used_seams = True
        # Швы по обеим осям — это сетка кадров: режем сразу на ячейки. Иначе — по более сильной оси.
        if seams_x and seams_y:
            cut_x, cut_y = [0, *seams_x, trimmed.w], [0, *seams_y, trimmed.h]
        elif (seams_y and score_y >= score_x) or not seams_x:
            cut_x, cut_y = [0, trimmed.w], [0, *seams_y, trimmed.h]
        else:
            cut_x, cut_y = [0, *seams_x, trimmed.w], [0, trimmed.h]
        self.seam_lines += [("x", trimmed.x1 + c, trimmed.y1, trimmed.y2) for c in cut_x[1:-1]]
        self.seam_lines += [("y", trimmed.y1 + c, trimmed.x1, trimmed.x2) for c in cut_y[1:-1]]
        parts = []
        for ya, yb in zip(cut_y, cut_y[1:]):
            for xa, xb in zip(cut_x, cut_x[1:]):
                part = Box(trimmed.x1 + xa, trimmed.y1 + ya, trimmed.x1 + xb, trimmed.y1 + yb)
                part_hard = _Hard(
                    left=hard.left if xa == 0 else True,
                    top=hard.top if ya == 0 else True,
                    right=hard.right if xb == trimmed.w else True,
                    bottom=hard.bottom if yb == trimmed.h else True,
                )
                parts.append((part, part_hard))
        return [leaf for part, part_hard in parts for leaf in self.split(part, part_hard, depth + 1)]


def _reading_order(boxes: list[Box]) -> list[Box]:
    """Сверху вниз по рядам, в ряду — слева направо."""
    if not boxes:
        return []
    median_h = float(np.median([b.h for b in boxes]))
    rows: list[list[Box]] = []
    for box in sorted(boxes, key=lambda b: (b.y1 + b.y2) / 2):
        center = (box.y1 + box.y2) / 2
        if rows and center - (rows[-1][0].y1 + rows[-1][0].y2) / 2 <= 0.5 * median_h:
            rows[-1].append(box)
        else:
            rows.append([box])
    return [box for row in rows for box in sorted(row, key=lambda b: b.x1)]


def _overlap(a1: int, a2: int, b1: int, b2: int) -> bool:
    return min(a2, b2) > max(a1, b1)


def _pad(boxes: list[Box], width: int, height: int) -> list[Box]:
    """Небольшой запас по краям, но не больше половины расстояния до соседнего кадра."""
    padded = []
    for box in boxes:
        pad = round(PAD_FRAC * max(box.w, box.h))
        left, top, right, bottom = min(pad, box.x1), min(pad, box.y1), min(pad, width - box.x2), min(pad, height - box.y2)
        for other in boxes:
            if other is box:
                continue
            if _overlap(box.y1, box.y2, other.y1, other.y2):
                if other.x2 <= box.x1:
                    left = min(left, (box.x1 - other.x2) // 2)
                if other.x1 >= box.x2:
                    right = min(right, (other.x1 - box.x2) // 2)
            if _overlap(box.x1, box.x2, other.x1, other.x2):
                if other.y2 <= box.y1:
                    top = min(top, (box.y1 - other.y2) // 2)
                if other.y1 >= box.y2:
                    bottom = min(bottom, (other.y1 - box.y2) // 2)
        padded.append(Box(box.x1 - left, box.y1 - top, box.x2 + right, box.y2 + bottom))
    return padded


def _snap_to_seams(views: list[Box], rgb: Image.Image, seam_lines: list[tuple[str, int, int, int]], scale: float) -> list[Box]:
    """Уточняет края кадров у швов по оригиналу (до пикселя) и отрезает смешанные пиксели шва."""
    if not seam_lines:
        return views
    a = np.asarray(rgb, dtype=np.float32)
    height, width = a.shape[:2]
    radius = int(np.ceil(SEAM_SNAP_RADIUS * scale)) + 1
    inset = max(SEAM_INSET_PX, round(scale))
    edges = {id(v): [v.x1, v.y1, v.x2, v.y2] for v in views}
    for axis, pos, lo, hi in seam_lines:
        center, lo, hi = round(pos * scale), int(lo * scale), int(np.ceil(hi * scale))
        limit = width if axis == "x" else height
        candidates = range(max(1, center - radius), min(limit - 1, center + radius) + 1)
        if axis == "x":
            strip = a[lo:hi]
            steps = [float(np.median(np.abs(strip[:, b] - strip[:, b - 1]).max(axis=1))) for b in candidates]
        else:
            strip = a[:, lo:hi]
            steps = [float(np.median(np.abs(strip[b] - strip[b - 1]).max(axis=1))) for b in candidates]
        boundary = candidates[int(np.argmax(steps))]
        for v in views:
            e = edges[id(v)]
            near, far = (1, 3) if axis == "x" else (0, 2)  # индексы поперечных краёв: (y1, y2) или (x1, x2)
            first, last = (0, 2) if axis == "x" else (1, 3)  # края вдоль оси шва
            if not _overlap(e[near], e[far], lo, hi):
                continue
            if abs(e[last] - center) <= radius + 1:
                e[last] = boundary - inset
            if abs(e[first] - center) <= radius + 1:
                e[first] = boundary + inset
    return [Box(*edges[id(v)]) for v in views]


def find_views(img: Image.Image) -> Layout:
    """Находит кадры на картинке. Координаты — в пикселях оригинала."""
    rgb = img.convert("RGB")
    width, height = rgb.size
    scale = max(width, height) / ANALYSIS_MAX_EDGE
    if scale > 1:
        small = rgb.resize((max(1, round(width / scale)), max(1, round(height / scale))), Image.Resampling.BOX)
    else:
        scale, small = 1.0, rgb
    raw = np.asarray(small, dtype=np.float32)
    smooth = np.asarray(small.filter(ImageFilter.BoxBlur(1)), dtype=np.float32)

    sw, sh = small.size
    splitter = _Splitter(raw, smooth)
    leaves = splitter.split(Box(0, 0, sw, sh))
    if not leaves:  # картинка пустая — один кадр на всю площадь
        whole = Box(0, 0, width, height)
        return Layout(views=[whole], content=whole, method="пустая картинка", dropped=0)

    largest = max(b.area for b in leaves)

    def is_noise(b: Box) -> bool:
        strip = max(b.w, b.h) >= STRIP_ASPECT * min(b.w, b.h) and b.area < STRIP_MAX_AREA_FRAC * largest
        return b.area < MIN_AREA_FRAC * largest or strip

    kept = [b for b in leaves if not is_noise(b)]
    dropped = len(leaves) - len(kept)
    kept = _reading_order(kept)[:MAX_VIEWS]

    content = Box(min(b.x1 for b in kept), min(b.y1 for b in kept), max(b.x2 for b in kept), max(b.y2 for b in kept))
    padded = _pad(kept, sw, sh)

    def to_original(b: Box) -> Box:
        return Box(
            max(0, int(b.x1 * scale)),
            max(0, int(b.y1 * scale)),
            min(width, int(np.ceil(b.x2 * scale))),
            min(height, int(np.ceil(b.y2 * scale))),
        )

    if len(kept) == 1:
        method = "один кадр"
    elif splitter.used_gaps and splitter.used_seams:
        method = "промежутки и швы"
    elif splitter.used_seams:
        method = "швы (кадры встык)"
    else:
        method = "промежутки"
    views = _snap_to_seams([to_original(b) for b in padded], rgb, splitter.seam_lines, scale)
    return Layout(views=views, content=to_original(content), method=method, dropped=dropped)
