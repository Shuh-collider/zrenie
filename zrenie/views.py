"""Набор видов: откуда взять вид (файл, кадр листа, область), замеры силуэта, выравнивание и экспорт."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import draw
from .layout import Box, Layout
from .matting import Matte

# Вид задаётся строкой: "путь", "путь|3" (кадр 3 листа из inspect), "путь|0" (вся картинка, даже если
# сервер счёл её листом) или "путь|x1,y1,x2,y2" (область). В конце можно добавить имя детали из parts:
# "путь|чашка", "путь|3|чашка" — тогда вместо силуэта всего объекта берётся маска этой детали.
# Символ | в именах файлов Windows запрещён, поэтому путь он не испортит.
SPEC_SEPARATOR = "|"

# Подписи видов: русские и английские варианты → одно имя.
LABEL_ALIASES = {
    "front": "front", "спереди": "front", "перед": "front", "фронт": "front",
    "back": "back", "rear": "back", "сзади": "back", "зад": "back",
    "left": "left", "слева": "left", "лево": "left",
    "right": "right", "справа": "right", "право": "right",
    "side": "side", "сбоку": "side", "бок": "side", "профиль": "side",
    "top": "top", "сверху": "top", "верх": "top",
    "bottom": "bottom", "снизу": "bottom", "низ": "bottom",
    "3-4": "three_quarter", "3/4": "three_quarter", "three_quarter": "three_quarter", "34": "three_quarter",
}
LABEL_NAMES = {
    "front": "спереди", "back": "сзади", "left": "слева", "right": "справа", "side": "сбоку",
    "top": "сверху", "bottom": "снизу", "three_quarter": "3/4",
}
PLAN_VIEWS = {"top", "bottom"}  # вертикаль картинки у них — глубина объекта, а не высота

ALIGN_HEIGHT = 900  # высота объекта на листе выравнивания до ужатия под лимиты Claude
ALIGN_MARGIN = 40
ALIGN_SCALE_W = 70  # справа — подписи высот в процентах
ALIGN_GAP = 60
ALIGN_LABEL_H = 50
ALIGN_BACKGROUND = (128, 128, 128)
GUIDE_COLOUR = (255, 0, 255)
KEY_HEIGHTS = 6
KEY_MIN_SEPARATION = 0.06  # ключевые высоты не ближе 6 % высоты друг к другу
KEY_MIN_CHANGE = 0.08  # скачок ширины и заполнения меньше 8 % ширины объекта — не ключевая высота
KEY_MATCH_TOL = 0.08  # высота у другого вида — та же деталь, если ближе 8 %
KEY_REPORT_DIFF = 0.02  # расхождение меньше 2 % не сообщаем
PAIR_REPORT_DIFF = 0.03
EXPORT_MAX_HEIGHT = 2048
EXPORT_PAD_FRAC = 0.04


class ViewError(Exception):
    """Вид нельзя взять; текст уходит модели."""


@dataclass
class ViewSource:
    spec: str
    path: Path
    box: Box  # область в пикселях оригинала
    image: Image.Image  # вырезанная область
    name: str
    sheet_views: int  # сколько кадров на исходной картинке
    part: str | None = None  # имя детали из parts — маска детали вместо силуэта объекта


def _coords(selector: str) -> tuple[int, int, int, int] | None:
    values = selector.strip("[]() ").split(",")
    if len(values) != 4:
        return None
    try:
        x1, y1, x2, y2 = (round(float(v)) for v in values)
    except ValueError:
        return None
    return x1, y1, x2, y2


def is_frame_selector(selector: str) -> bool:
    """«3», «0» или «x1,y1,x2,y2» (всё с запятыми) — выбор кадра или области; остальное — имя детали."""
    return selector.isdigit() or "," in selector


def normalize_label(label: str | None) -> str | None:
    if not label:
        return None
    return LABEL_ALIASES.get(label.strip().lower())


def label_text(label: str | None, fallback: str) -> str:
    canonical = normalize_label(label)
    if canonical:
        return LABEL_NAMES[canonical]
    return label or fallback


def parse_spec(
    spec: str,
    open_image: Callable[[str], tuple[Image.Image, Path, tuple]],
    get_layout: Callable[[tuple], Layout],
    require_single: bool = True,
) -> ViewSource:
    """Разбирает «путь», «путь|N», «путь|x1,y1,x2,y2» и то же с деталью в конце: «путь|деталь», «путь|N|деталь»."""
    path_part, *selectors = spec.split(SPEC_SEPARATOR)
    selectors = [s.strip() for s in selectors]
    if len(selectors) > 2 or (len(selectors) == 2 and not is_frame_selector(selectors[0])):
        raise ViewError(f"Не понял вид «{spec}»: нужно «путь», «путь|N», «путь|x1,y1,x2,y2», можно с деталью в конце: «путь|N|деталь».")
    selector, part = "", None
    if len(selectors) == 2:
        selector, part = selectors
    elif selectors:
        if is_frame_selector(selectors[0]):
            selector = selectors[0]
        else:
            part = selectors[0]
    if part is not None and not part:
        raise ViewError(f"Пустое имя детали в «{spec}».")
    img, p, key = open_image(path_part.strip())
    layout = get_layout(key)
    count = len(layout.views)

    if not selector:
        if require_single and count > 1:
            raise ViewError(
                f"{p.name} похожа на лист из {count} кадров. Укажи кадр: «{path_part.strip()}|1» … «|{count}» "
                f"(номера из inspect) или «|0», если это один объект целиком (например, из отдельных частей)."
                + (f" С деталью: «{path_part.strip()}|N|{part}»." if part else "")
            )
        box, name = Box(0, 0, img.width, img.height), p.name
    elif selector == "0":
        box, name = Box(0, 0, img.width, img.height), p.name
    elif selector.isdigit():
        n = int(selector)
        if not 1 <= n <= count:
            raise ViewError(f"На {p.name} кадров {count}, а запрошен кадр {n}.")
        box, name = layout.views[n - 1], f"{p.name}, кадр {n}"
    else:
        coords = _coords(selector)
        if coords is None:
            raise ViewError(f"Не понял область «{selector}»: нужно x1,y1,x2,y2 в пикселях оригинала.")
        x1, y1, x2, y2 = coords
        x1, x2 = sorted((max(0, x1), min(img.width, x2)))
        y1, y2 = sorted((max(0, y1), min(img.height, y2)))
        if x2 - x1 < 16 or y2 - y1 < 16:
            raise ViewError(f"Область «{selector}» вне картинки или слишком мала ({img.width}×{img.height} px).")
        box, name = Box(x1, y1, x2, y2), f"{p.name} [{x1}, {y1}, {x2}, {y2}]"
    if part:
        name += f", деталь «{part}»"
    return ViewSource(spec=spec, path=p, box=box, image=img.crop(box.as_list()), name=name, sheet_views=count, part=part)


@dataclass
class Shape:
    bbox: Box  # рамка объекта в координатах вида (вырезанной области)
    area: int
    parts: int
    holes: int
    centroid: tuple[float, float]
    key_heights: list[float]  # доли высоты от низа (0) до верха (1), где резко меняется ширина силуэта

    @property
    def aspect(self) -> float:
        return self.bbox.w / self.bbox.h


def _key_heights(mask: np.ndarray, bbox: Box) -> list[float]:
    """Высоты, где резко меняется ширина силуэта или сколько в строке «объекта» (начало и конец деталей)."""
    rows = mask[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
    h = rows.shape[0]
    if h < 8:
        return []
    any_px = rows.any(axis=1)
    first = np.argmax(rows, axis=1)
    last = rows.shape[1] - 1 - np.argmax(rows[:, ::-1], axis=1)
    extent = np.where(any_px, last - first + 1, 0).astype(np.float32)
    filled = rows.sum(axis=1).astype(np.float32)
    window = max(3, h // 100)
    kernel = np.ones(window) / window
    change = np.abs(np.diff(np.convolve(extent, kernel, mode="same"))) + np.abs(np.diff(np.convolve(filled, kernel, mode="same")))
    # Резкий скачок даёт большое изменение на нескольких строках подряд — берём сумму по окну, а не одну строку.
    change = np.convolve(change, np.ones(window), mode="same")
    picked: list[int] = []
    for r in np.argsort(-change):
        if change[r] < KEY_MIN_CHANGE * bbox.w or len(picked) >= KEY_HEIGHTS:
            break
        if all(abs(r - q) >= KEY_MIN_SEPARATION * h for q in picked) and KEY_MIN_SEPARATION * h <= r <= (1 - KEY_MIN_SEPARATION) * h:
            picked.append(int(r))
    return sorted(((h - (r + 1)) / h for r in picked), reverse=True)


def measure(matte: Matte) -> Shape | None:
    mask = matte.mask
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    bbox = Box(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    area = int(mask.sum())
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    parts = int((stats[1:, cv2.CC_STAT_AREA] >= 0.01 * area).sum())
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        for contour, (_, _, _, parent) in zip(contours, hierarchy[0]):
            if parent != -1 and cv2.contourArea(contour) >= 0.001 * area:
                holes += 1
    return Shape(bbox, area, parts, holes, (float(xs.mean()), float(ys.mean())), _key_heights(mask, bbox))


def silhouette_preview(source: ViewSource, matte: Matte, shape: Shape | None) -> Image.Image:
    """Объект на нейтральном сером, фон бледный, контур маски пурпурным, рамка объекта — тонкая."""
    rgb = np.asarray(source.image.convert("RGB")).astype(np.float32)
    alpha = matte.alpha[..., None]
    faded = rgb * 0.25 + np.array(draw.NEUTRAL_LIGHT, np.float32) * 0.75
    out = (rgb * alpha + faded * (1 - alpha)).astype(np.uint8)
    contours, _ = cv2.findContours(matte.mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    thickness = max(1, round(max(out.shape[:2]) / 500))
    cv2.drawContours(out, contours, -1, GUIDE_COLOUR, thickness)
    if shape:
        b = shape.bbox
        cv2.rectangle(out, (b.x1, b.y1), (b.x2 - 1, b.y2 - 1), (0, 160, 255), max(1, thickness // 2))
    return Image.fromarray(out)


@dataclass
class AlignedView:
    source: ViewSource
    matte: Matte
    shape: Shape
    label: str | None  # каноническое имя или None
    title: str
    scale: float  # во сколько раз увеличен (уменьшен) вид на листе
    placed: Box | None = None  # где объект лежит на листе


def _cutout(view: AlignedView, height: int | None = None, scale: float | None = None, pad_frac: float = 0.0) -> Image.Image:
    """Объект по рамке (с запасом), RGBA с прозрачным фоном, в нужном масштабе."""
    b = view.shape.bbox
    pad = round(pad_frac * max(b.w, b.h))
    x1, y1 = max(0, b.x1 - pad), max(0, b.y1 - pad)
    x2, y2 = min(view.source.image.width, b.x2 + pad), min(view.source.image.height, b.y2 + pad)
    rgba = view.source.image.convert("RGB").crop((x1, y1, x2, y2)).convert("RGBA")
    rgba.putalpha(Image.fromarray((view.matte.alpha[y1:y2, x1:x2] * 255).astype(np.uint8)))
    s = scale if scale is not None else height / b.h
    size = (max(1, round(rgba.width * s)), max(1, round(rgba.height * s)))
    return rgba.resize(size, Image.Resampling.LANCZOS)


def plan_scales(views: list[AlignedView], target_height: float) -> None:
    """Высоту объекта у всех видов делаем одинаковой. Вид сверху (снизу) — по ширине вида спереди или сзади."""
    elevations = [v for v in views if v.label not in PLAN_VIEWS]
    for v in elevations:
        v.scale = target_height / v.shape.bbox.h
    reference = next((v for v in elevations if v.label in ("front", "back")), elevations[0] if elevations else None)
    for v in views:
        if v.label in PLAN_VIEWS:
            v.scale = (reference.shape.bbox.w * reference.scale / v.shape.bbox.w) if reference else target_height / v.shape.bbox.h


def compose_aligned(views: list[AlignedView]) -> tuple[Image.Image, list[float]]:
    """Лист: объекты одной высоты стоят на общей линии низа, направляющие — на ключевых высотах."""
    plan_scales(views, ALIGN_HEIGHT)
    cutouts = [_cutout(v, scale=v.scale) for v in views]
    tallest = max(c.height for c in cutouts)
    width = 2 * ALIGN_MARGIN + ALIGN_SCALE_W + sum(c.width for c in cutouts) + ALIGN_GAP * (len(cutouts) - 1)
    height = 2 * ALIGN_MARGIN + ALIGN_LABEL_H + tallest
    canvas = Image.new("RGB", (width, height), ALIGN_BACKGROUND)
    baseline = ALIGN_MARGIN + ALIGN_LABEL_H + tallest
    top = baseline - ALIGN_HEIGHT

    reference = next((v for v in views if v.label not in PLAN_VIEWS), None)
    guides = reference.shape.key_heights if reference else []
    pen = ImageDraw.Draw(canvas)
    small = draw.font(20)
    for level in (0.0, 1.0, *guides):
        y = round(baseline - level * ALIGN_HEIGHT)
        dash = 1.0 if level in (0.0, 1.0) else 0.5
        for x in range(0, width, 16):
            pen.line([(x, y), (x + round(16 * dash) - 1, y)], fill=GUIDE_COLOUR, width=2)
        pen.text((width - 4, y - 2), f"{level * 100:.0f}%", fill=GUIDE_COLOUR, font=small, anchor="rb")

    x = ALIGN_MARGIN
    label_font = draw.font(26)
    for v, cut in zip(views, cutouts):
        y = baseline - cut.height
        canvas.paste(cut, (x, y), cut)
        v.placed = Box(x, y, x + cut.width, baseline)
        pen.text((x, ALIGN_MARGIN + ALIGN_LABEL_H // 2), v.title, fill="white", font=label_font, anchor="lm")
        x += cut.width + ALIGN_GAP
    return canvas, guides


def _pct(a: float, b: float) -> float:
    return (a - b) / b * 100


def consistency_report(views: list[AlignedView]) -> list[str]:
    """Несостыковки между видами: по парам видов, по высотам с одного листа, по ключевым высотам."""
    lines: list[str] = []
    by_label = {v.label: v for v in views if v.label}

    def width_at_common_height(v: AlignedView) -> float:
        return v.shape.aspect  # ширина при высоте 1

    pairs = [("front", "back", "ширина спереди и сзади"), ("left", "right", "ширина слева и справа")]
    for a, b, what in pairs:
        if a in by_label and b in by_label:
            diff = _pct(width_at_common_height(by_label[b]), width_at_common_height(by_label[a]))
            if abs(diff) > PAIR_REPORT_DIFF * 100:
                lines.append(f"{what} не совпадает: у «{by_label[b].title}» шире на {diff:+.1f} % при той же высоте.")
    face = by_label.get("front") or by_label.get("back")
    side = by_label.get("side") or by_label.get("left") or by_label.get("right")
    plan = by_label.get("top") or by_label.get("bottom")
    if plan and side:
        depth_plan = plan.shape.bbox.h * plan.scale
        depth_side = side.shape.bbox.w * side.scale
        diff = _pct(depth_plan, depth_side)
        if abs(diff) > PAIR_REPORT_DIFF * 100:
            lines.append(f"Глубина по виду «{plan.title}» отличается от вида «{side.title}» на {diff:+.1f} %.")
    if plan and not face:
        lines.append(f"Вид «{plan.title}» масштабирован по своей высоте: нет вида спереди или сзади, чтобы подогнать по ширине.")

    # Виды с одного листа нарисованы в одном масштабе — их высоты можно сравнить напрямую.
    by_file: dict[Path, list[AlignedView]] = {}
    for v in views:
        if v.source.sheet_views > 1 and v.label not in PLAN_VIEWS:
            by_file.setdefault(v.source.path, []).append(v)
    for group in by_file.values():
        first = group[0]
        for other in group[1:]:
            diff = _pct(other.shape.bbox.h, first.shape.bbox.h)
            if abs(diff) > PAIR_REPORT_DIFF * 100:
                lines.append(
                    f"«{other.title}» и «{first.title}» с одного листа, но объект выше на {diff:+.1f} % "
                    "(разный масштаб или разная камера в генерации)."
                )

    reference = next((v for v in views if v.label not in PLAN_VIEWS), None)
    if reference and reference.shape.key_heights:
        for v in views:
            if v is reference or v.label in PLAN_VIEWS or not v.shape.key_heights:
                continue
            for level in reference.shape.key_heights:
                nearest = min(v.shape.key_heights, key=lambda k: abs(k - level))
                if KEY_REPORT_DIFF < abs(nearest - level) <= KEY_MATCH_TOL:
                    lines.append(
                        f"Ключевая высота {level * 100:.0f} % у «{reference.title}» — у «{v.title}» на {nearest * 100:.0f} % "
                        "(по силуэту; проверь глазами, та ли это деталь)."
                    )
    return lines


def export(views: list[AlignedView], folder: Path, real_height: float | None) -> tuple[list[Path], Path, dict]:
    """PNG с прозрачным фоном одной высоты объекта + views.json с масштабами для подложек в Blender / C4D."""
    folder.mkdir(parents=True, exist_ok=True)
    target = min(EXPORT_MAX_HEIGHT, max(v.shape.bbox.h for v in views if v.label not in PLAN_VIEWS) if any(v.label not in PLAN_VIEWS for v in views) else max(v.shape.bbox.h for v in views))
    plan_scales(views, target)
    units_per_px = real_height / target if real_height else None

    files, records = [], []
    used: set[str] = set()
    for i, v in enumerate(views, 1):
        stem = f"{i:02d}_{v.label or 'view'}"
        while stem in used:
            stem += "_"
        used.add(stem)
        cut = _cutout(v, scale=v.scale, pad_frac=EXPORT_PAD_FRAC)
        path = folder / f"{stem}.png"
        cut.save(path)
        files.append(path)

        b = v.shape.bbox
        pad = round(EXPORT_PAD_FRAC * max(b.w, b.h))
        crop_x1, crop_y1 = max(0, b.x1 - pad), max(0, b.y1 - pad)
        # Где объект на сохранённой картинке (в её пикселях).
        ox1, oy1 = (b.x1 - crop_x1) * v.scale, (b.y1 - crop_y1) * v.scale
        ox2, oy2 = ox1 + b.w * v.scale, oy1 + b.h * v.scale
        bottom_center = ((ox1 + ox2) / 2, oy2)
        record = {
            "file": path.name,
            "label": v.label,
            "title": v.title,
            "source": str(v.source.path),
            "source_box": v.source.box.as_list(),
            "object_box_in_source": [v.source.box.x1 + b.x1, v.source.box.y1 + b.y1, v.source.box.x1 + b.x2, v.source.box.y1 + b.y2],
            "image_px": [cut.width, cut.height],
            "object_box_px": [round(ox1, 1), round(oy1, 1), round(ox2, 1), round(oy2, 1)],
            "object_bottom_center_px": [round(bottom_center[0], 1), round(bottom_center[1], 1)],
            "scale_from_source": round(v.scale, 4),
        }
        if units_per_px:
            # Смещение центра картинки от низа-центра объекта: x вправо, вверх — положительно.
            record["image_size_units"] = [round(cut.width * units_per_px, 4), round(cut.height * units_per_px, 4)]
            record["image_center_from_object_bottom_center_units"] = [
                round((cut.width / 2 - bottom_center[0]) * units_per_px, 4),
                round((bottom_center[1] - cut.height / 2) * units_per_px, 4),
            ]
        records.append(record)

    meta = {
        "object_height_px": target,
        "real_height_units": real_height,
        "units_per_px": units_per_px,
        "note": "Все виды (кроме вида сверху/снизу) приведены к одной высоте объекта. Вид сверху/снизу подогнан по ширине вида спереди/сзади.",
        "views": records,
    }
    meta_path = folder / "views.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return files, meta_path, meta


def fmt_ratio(value: float) -> str:
    return f"{value:.3f}" if not math.isnan(value) else "—"
