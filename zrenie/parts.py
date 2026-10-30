"""Детали объекта: выделение по точкам и рамкам (SAM 2.1), замеры деталей, детали модели по «карте деталей».

SAM 2.1 (Apache 2.0) выделяет деталь, на которую указали точками или рамкой; сам названий не знает — точки
ставит Claude. «Карта деталей» — рендер, где каждый объект сцены залит своим плоским цветом (Blender:
blender_kit.render_parts), рядом JSON «объект → цвет»; по ней детали модели вырезаются точно, без нейросети.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import draw
from .layout import Box
from .matting import GPU_LOCK

log = logging.getLogger(__name__)

SAM_REPO = "facebook/sam2.1-hiera-large"
SAM_REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
MIN_PIECE_FRAC = 0.01  # куски детали меньше 1 % её самой большой части — шум
ELONGATED = 1.15  # длинная ось хотя бы на 15 % длиннее короткой — наклон имеет смысл
OVERLAP_REPORT = 0.2  # детали перекрываются больше чем на 20 % меньшей — сказать
WHOLE_OBJECT = 0.85  # деталь больше 85 % объекта — скорее всего, выделился весь объект
OUTSIDE_REPORT = 0.15  # больше 15 % детали вне силуэта объекта — подозрительно
# Сравнение деталей: о меньших расхождениях не сообщаем.
SIZE_REPORT = 3.0  # % размера рамки детали
SHIFT_REPORT = 1.0  # % размера объекта
TILT_ELONGATED = 1.4  # наклон сравниваем, только если обе детали заметно вытянуты
TILT_REPORT = 3.0  # градусов
ELONGATION_REPORT = 0.2  # вытянутость отличается больше чем на 20 %

# Цвета деталей (они же — цвета объектов на карте деталей Blender). Между любыми двумя — заметная разница.
COLOURS = [
    ((230, 25, 75), "красный"),
    ((60, 180, 75), "зелёный"),
    ((0, 130, 200), "синий"),
    ((245, 130, 48), "оранжевый"),
    ((145, 30, 180), "фиолетовый"),
    ((70, 240, 240), "голубой"),
    ((240, 50, 230), "пурпурный"),
    ((210, 245, 60), "салатовый"),
    ((0, 128, 128), "бирюзовый"),
    ((170, 110, 40), "коричневый"),
    ((128, 0, 0), "бордовый"),
    ((0, 0, 128), "тёмно-синий"),
    ((255, 225, 25), "жёлтый"),
    ((128, 128, 0), "оливковый"),
    ((250, 190, 212), "розовый"),
    ((170, 255, 195), "мятный"),
    ((220, 190, 255), "сиреневый"),
    ((255, 215, 180), "персиковый"),
    ((128, 128, 128), "серый"),
    ((255, 250, 200), "кремовый"),
]
ID_TOLERANCE = 12  # цвет пикселя карты деталей отличается от заданного не больше чем на столько (RGB)


@dataclass
class Prompt:
    name: str
    points: list[tuple[float, float]] = field(default_factory=list)  # на детали; координаты вида
    exclude: list[tuple[float, float]] = field(default_factory=list)  # не на детали
    box: tuple[float, float, float, float] | None = None
    pick: str = "auto"  # для одной точки: auto — самая уверенная из трёх масок, small / medium / large — по размеру


PICKS = ("auto", "small", "medium", "large")


@dataclass
class Part:
    name: str
    mask: np.ndarray  # bool, H×W вида
    score: float  # уверенность SAM 0..1; у деталей модели — 1
    prompt: Prompt | None = None
    options: list[tuple[str, int, float]] = field(default_factory=list)  # для одной точки: (размер, площадь, уверенность)


# ---------- SAM 2.1 ----------

_model = None
_processor = None
_device = "cpu"


def _load():
    global _model, _processor, _device
    if _model is None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        log.info("Загружаю SAM 2.1…")
        try:  # обычно модель уже на диске — в интернет не ходим
            processor = Sam2Processor.from_pretrained(SAM_REPO, revision=SAM_REVISION, local_files_only=True)
            model = Sam2Model.from_pretrained(SAM_REPO, revision=SAM_REVISION, local_files_only=True)
        except OSError:  # первый запуск: скачать ~0,9 ГБ
            processor = Sam2Processor.from_pretrained(SAM_REPO, revision=SAM_REVISION)
            model = Sam2Model.from_pretrained(SAM_REPO, revision=SAM_REVISION)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(_device).eval()
        _model, _processor = model, processor
        log.info("SAM 2.1 загружен на %s", _device)
    return _model, _processor


def device_name() -> str:
    return "видеокарта" if _device == "cuda" else "процессор"


def _keep_main(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(count, bool)
    keep[1:] = areas >= MIN_PIECE_FRAC * areas.max()
    return keep[labels]


def segment(img: Image.Image, prompts: list[Prompt]) -> list[Part]:
    """Маска для каждой подсказки. Картинка кодируется один раз, подсказки — по очереди."""
    import torch

    with GPU_LOCK:
        model, processor = _load()
        rgb = img.convert("RGB")
        base = processor(images=rgb, return_tensors="pt")
        sizes = base["original_sizes"]
        result = []
        try:
            with torch.no_grad():
                embeddings = model.get_image_embeddings(base["pixel_values"].to(_device))
                for p in prompts:
                    kwargs = {}
                    points = [*p.points, *p.exclude]
                    if points:
                        kwargs["input_points"] = [[[list(map(float, q)) for q in points]]]
                        kwargs["input_labels"] = [[[1] * len(p.points) + [0] * len(p.exclude)]]
                    if p.box is not None:
                        kwargs["input_boxes"] = [[list(map(float, p.box))]]
                    inputs = processor(original_sizes=sizes, return_tensors="pt", **kwargs)
                    inputs = {k: v.to(_device) for k, v in inputs.items() if k != "original_sizes"}
                    # Одна точка — неоднозначна (деталь, её часть или весь объект): берём лучшую из трёх масок.
                    multimask = len(p.points) == 1 and not p.exclude and p.box is None
                    out = model(image_embeddings=embeddings, multimask_output=multimask, **inputs)
                    logits = processor.post_process_masks(out.pred_masks.cpu(), sizes, binarize=False)[0][0]  # (масок, H, W)
                    scores = out.iou_scores[0, 0].float().cpu().numpy()
                    masks = [_keep_main(m.float().numpy() > 0) for m in logits]
                    options = []
                    if multimask:
                        by_size = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()))
                        sizes_named = dict(zip(("small", "medium", "large"), by_size))
                        options = [(size, int(masks[i].sum()), float(scores[i])) for size, i in sizes_named.items()]
                        best = sizes_named[p.pick] if p.pick in sizes_named else int(np.argmax(scores))
                    else:
                        best = int(np.argmax(scores))
                    result.append(Part(p.name, masks[best], float(scores[best]), p, options))
        finally:
            if _device == "cuda":
                torch.cuda.empty_cache()  # видеопамять нужна Blender
    return result


# ---------- замеры детали ----------


@dataclass
class PartShape:
    bbox: Box
    area: int
    pieces: int
    holes: int
    centroid: tuple[float, float]
    angle: float | None  # наклон длинной оси к горизонтали, против часовой, −90…90; None — деталь не вытянута
    elongation: float  # длинная ось / короткая
    colour: str  # медианный цвет детали, HEX


def measure_part(mask: np.ndarray, image: Image.Image | None = None) -> PartShape | None:
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    bbox = Box(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    area = int(mask.sum())
    m8 = mask.astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    pieces = int((stats[1:, cv2.CC_STAT_AREA] >= MIN_PIECE_FRAC * area).sum())
    contours, hierarchy = cv2.findContours(m8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        holes = sum(1 for c, h in zip(contours, hierarchy[0]) if h[3] != -1 and cv2.contourArea(c) >= 0.002 * area)
    mo = cv2.moments(m8, binaryImage=True)
    mu20, mu02, mu11 = mo["mu20"] / mo["m00"], mo["mu02"] / mo["m00"], mo["mu11"] / mo["m00"]
    spread = math.sqrt(((mu20 - mu02) / 2) ** 2 + mu11**2)
    major, minor = (mu20 + mu02) / 2 + spread, (mu20 + mu02) / 2 - spread
    elongation = math.sqrt(major / minor) if minor > 1e-9 else float("inf")
    angle = None
    if elongation >= ELONGATED:
        theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)  # в координатах картинки (y вниз)
        angle = -math.degrees(theta)  # против часовой при y вверх
        angle = (angle + 90) % 180 - 90
    colour = "—"
    if image is not None:
        rgb = np.asarray(image.convert("RGB"))[mask]
        r, g, b = np.median(rgb, axis=0).astype(int)
        colour = f"#{r:02X}{g:02X}{b:02X}"
    return PartShape(bbox, area, pieces, holes, (float(xs.mean()), float(ys.mean())), angle, elongation, colour)


def angle_text(angle: float | None, elongation: float) -> str:
    if angle is None:
        return "длинной оси нет (деталь почти круглая или квадратная)"
    return f"длинная ось под {angle:.1f}° к горизонтали ({90 - abs(angle):.1f}° от вертикали), вытянутость {elongation:.2f}"


def relative_text(s: PartShape, obj: Box, origin: tuple[int, int]) -> str:
    """Размер и место детали в процентах рамки всего объекта. obj — рамка объекта в координатах вида."""
    x1 = (s.bbox.x1 - obj.x1) / obj.w * 100
    x2 = (s.bbox.x2 - obj.x1) / obj.w * 100
    z1 = (obj.y2 - s.bbox.y2) / obj.h * 100
    z2 = (obj.y2 - s.bbox.y1) / obj.h * 100
    b = s.bbox
    return (
        f"рамка в оригинале [{origin[0] + b.x1}, {origin[1] + b.y1}, {origin[0] + b.x2}, {origin[1] + b.y2}] — {b.w}×{b.h} px; "
        f"от объекта: ширина {b.w / obj.w * 100:.1f} %, высота {b.h / obj.h * 100:.1f} %; "
        f"место: x {x1:.1f}–{x2:.1f} % ширины слева, высота {z1:.1f}–{z2:.1f} % от низа"
    )


def warnings(parts: list[Part], obj_mask: np.ndarray | None) -> list[str]:
    """Подозрительное: перекрытия деталей, деталь размером с весь объект, деталь вне силуэта."""
    lines = []
    obj_area = int(obj_mask.sum()) if obj_mask is not None else 0
    for p in parts:
        area = int(p.mask.sum())
        if not area:
            lines.append(f"«{p.name}»: SAM ничего не выделил — поставь точку внутрь детали или задай box.")
            continue
        if obj_area and area >= WHOLE_OBJECT * obj_area:
            lines.append(f"«{p.name}»: выделен почти весь объект ({area / obj_area * 100:.0f} %) — добавь точки или box, исключи соседние детали (exclude).")
        if obj_area:
            outside = int((p.mask & ~obj_mask).sum()) / area
            if outside >= OUTSIDE_REPORT:
                lines.append(f"«{p.name}»: {outside * 100:.0f} % детали вне силуэта объекта — захвачен фон или тень.")
    for i, a in enumerate(parts):
        for b in parts[i + 1 :]:
            small = min(int(a.mask.sum()), int(b.mask.sum()))
            if small and int((a.mask & b.mask).sum()) / small >= OVERLAP_REPORT:
                lines.append(f"«{a.name}» и «{b.name}» перекрываются на {int((a.mask & b.mask).sum()) / small * 100:.0f} % меньшей — проверь точки.")
    return lines


# ---------- картинка с деталями ----------


def _label_point(mask: np.ndarray) -> tuple[int, int]:
    """Точка глубоко внутри детали — туда ставится номер."""
    dist = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(x), int(y)


def draw_parts(image: Image.Image, parts: list[Part], scale: float, numbers: list[int] | None = None, fill: float = 0.45) -> Image.Image:
    """Детали цветными заливками с контуром и номером; точки и рамки подсказок. scale — пикселей вида на пиксель картинки."""
    out = np.asarray(image.convert("RGB")).astype(np.float32)
    h, w = out.shape[:2]
    thickness = max(2, round(max(h, w) / 450))
    small = [cv2.resize(p.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) for p in parts]
    for i, m in enumerate(small):
        colour = np.array(COLOURS[(numbers[i] - 1 if numbers else i) % len(COLOURS)][0], np.float32)
        out[m] = out[m] * (1 - fill) + colour * fill
    out = out.astype(np.uint8)
    for i, m in enumerate(small):
        colour = COLOURS[(numbers[i] - 1 if numbers else i) % len(COLOURS)][0]
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, colour, thickness)
    picture = Image.fromarray(out)
    pen = ImageDraw.Draw(picture)
    label_font = draw.font(max(16, round(max(h, w) / 45)))
    r = max(4, thickness * 2)
    for i, (p, m) in enumerate(zip(parts, small)):
        colour = COLOURS[(numbers[i] - 1 if numbers else i) % len(COLOURS)][0]
        if p.prompt is not None:
            if p.prompt.box is not None:
                x1, y1, x2, y2 = (v / scale for v in p.prompt.box)
                pen.rectangle([x1, y1, x2, y2], outline=colour, width=max(1, thickness // 2))
            for x, y in p.prompt.points:
                pen.ellipse([x / scale - r, y / scale - r, x / scale + r, y / scale + r], fill="white", outline=colour, width=max(2, r // 2))
            for x, y in p.prompt.exclude:  # крестик цвета детали с чёрной обводкой — видно, чей он
                cx, cy = x / scale, y / scale
                for fill_, wd in (("black", max(3, r // 2 + 2)), (colour, max(1, r // 2 - 1))):
                    pen.line([cx - r, cy - r, cx + r, cy + r], fill=fill_, width=wd)
                    pen.line([cx - r, cy + r, cx + r, cy - r], fill=fill_, width=wd)
        if m.any():
            x, y = _label_point(m)
            text = str(numbers[i] if numbers else i + 1)
            left, top, right, bottom = pen.textbbox((0, 0), text, font=label_font)
            pad = 4
            pen.rectangle([x - (right - left) / 2 - pad, y - (bottom - top) / 2 - pad, x + (right - left) / 2 + pad, y + (bottom - top) / 2 + pad], fill=colour)
            pen.text((x, y), text, fill="white", font=label_font, anchor="mm")
    return picture


# ---------- детали модели по карте деталей ----------


def read_id_map(path: Path) -> dict[str, tuple[int, int, int]]:
    """JSON рядом с рендером: {"objects": {"имя объекта": "#RRGGBB", ...}}."""
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        raise ValueError(f"Нет {sidecar.name} рядом с рендером: это должна быть карта деталей из blender_kit.render_parts.")
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    return {name: _hex(value) for name, value in data.get("objects", {}).items()}


def _hex(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def model_parts(render: Image.Image, colours: dict[str, tuple[int, int, int]]) -> dict[str, np.ndarray]:
    """Маска каждого объекта по его цвету на карте деталей (прозрачное — фон)."""
    rgba = np.asarray(render.convert("RGBA")).astype(np.int16)
    visible = rgba[..., 3] >= 128
    result = {}
    for name, colour in colours.items():
        dist = np.abs(rgba[..., :3] - np.array(colour, np.int16)).max(axis=2)
        result[name] = visible & (dist <= ID_TOLERANCE)
    return result


# ---------- сравнение по деталям ----------


@dataclass
class PartComparison:
    name: str
    iou: float
    ref: PartShape | None
    model: PartShape | None
    findings: list[str]


def compare_part(name: str, ref_mask: np.ndarray, model_mask: np.ndarray, obj: Box) -> PartComparison:
    """Деталь референса против той же детали модели (маска модели уже в координатах референса)."""
    ref, model = measure_part(ref_mask), measure_part(model_mask)
    union = int((ref_mask | model_mask).sum())
    iou = int((ref_mask & model_mask).sum()) / union if union else 0.0
    if ref is None or model is None:
        what = "на референсе деталь не выделена" if ref is None else "у модели детали на этом ракурсе не видно"
        return PartComparison(name, iou, ref, model, [what])
    findings = []
    dw = (model.bbox.w - ref.bbox.w) / ref.bbox.w * 100
    dh = (model.bbox.h - ref.bbox.h) / ref.bbox.h * 100
    if abs(dw) >= SIZE_REPORT or abs(dh) >= SIZE_REPORT:
        findings.append(f"рамка детали: ширина {dw:+.1f} %, высота {dh:+.1f} %")
    dx = (model.centroid[0] - ref.centroid[0]) / obj.w * 100
    dz = (ref.centroid[1] - model.centroid[1]) / obj.h * 100
    shifts = []
    if abs(dx) >= SHIFT_REPORT:
        shifts.append(f"{'правее' if dx > 0 else 'левее'} на {abs(dx):.1f} % ширины объекта")
    if abs(dz) >= SHIFT_REPORT:
        shifts.append(f"{'выше' if dz > 0 else 'ниже'} на {abs(dz):.1f} % высоты объекта")
    if shifts:
        findings.append("центр " + " и ".join(shifts))
    if ref.elongation >= TILT_ELONGATED and model.elongation >= TILT_ELONGATED:
        da = (model.angle - ref.angle + 90) % 180 - 90
        if abs(da) >= TILT_REPORT:
            findings.append(
                f"наклон длинной оси {model.angle:.0f}° вместо {ref.angle:.0f}° к горизонтали "
                f"(от вертикали {90 - abs(model.angle):.0f}° вместо {90 - abs(ref.angle):.0f}°)"
            )
    if abs(model.elongation - ref.elongation) / ref.elongation >= ELONGATION_REPORT:
        findings.append(f"вытянутость {model.elongation:.2f} вместо {ref.elongation:.2f}")
    if model.holes != ref.holes:
        findings.append(f"отверстий {model.holes} вместо {ref.holes}")
    if model.pieces != ref.pieces:
        findings.append(f"кусков {model.pieces} вместо {ref.pieces}")
    return PartComparison(name, iou, ref, model, findings)


def compare_panels(
    ref_rgb: np.ndarray,
    ref_parts: list[np.ndarray],
    model_parts_: list[np.ndarray],
    model_mask: np.ndarray,
    crop: tuple[int, int, int, int],
    colour_ids: list[int],
) -> list[tuple[str, Image.Image]]:
    """Референс с деталями, модель (серый силуэт) с деталями тех же цветов и контуры деталей поверх друг друга.

    Всё — на холсте сравнения (compare.Comparison); crop — что показать: x1, y1, x2, y2.
    colour_ids — номер цвета каждой детали в COLOURS (как в parts)."""
    colours = [COLOURS[i % len(COLOURS)][0] for i in colour_ids]
    model_rgb = np.full_like(ref_rgb, 235)
    model_rgb[model_mask] = 150
    width = max(2, round(max(ref_rgb.shape[:2]) / 400))

    def filled(base: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
        out = base.astype(np.float32)
        for colour, m in zip(colours, masks):
            out[m] = out[m] * 0.45 + np.array(colour, np.float32) * 0.55
        out = out.astype(np.uint8)
        for colour, m in zip(colours, masks):
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, contours, -1, colour, width)
        return out

    faded = (ref_rgb.astype(np.float32) * 0.3 + 235 * 0.7).astype(np.uint8)
    lines = faded.copy()
    for colour, r, m in zip(colours, ref_parts, model_parts_):
        for mask, t in ((r, width + 1), (m, max(1, width // 2))):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(lines, contours, -1, colour, t)

    def cut(a: np.ndarray) -> Image.Image:
        return Image.fromarray(a[crop[1] : crop[3], crop[0] : crop[2]])

    return [
        ("референс", cut(filled(ref_rgb, ref_parts))),
        ("модель", cut(filled(model_rgb, model_parts_))),
        ("контуры: толстые — референс, тонкие — модель", cut(lines)),
    ]
