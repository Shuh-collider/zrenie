"""MCP-сервер «Зрение»: инструменты, которые вызывает Claude."""

import json
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from mcp.server import MCPServer
from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image as PILImage
from pydantic import BaseModel, Field

from . import __version__, compare as cmp, draw, imaging, layout, matting, measure, modes
from . import parts as pt
from . import views as vs

MAX_LIST = 300
MAX_SPLIT_SHOWN = 9
SPLIT_MAX_EDGE = 1024  # на кадр в split_views: 9 кадров укладываются в лимит ответа Claude Code
MAX_SHEET = 12
MAX_ALIGN = 8
MAX_COMPARE = 6
COMPARE_MODES = ("all", "side", "diff", "overlay")
MAX_PARTS = 12
LOW_SCORE = 0.8  # уверенность SAM ниже — маску стоит проверить глазами
BLENDER_KIT = Path(__file__).with_name("blender_kit.py")
BLENDER_HINT = (
    f'В Blender (execute_blender_code): exec(open(r"{BLENDER_KIT}", encoding="utf-8").read()); '
    'result = render_parts("<камера>", r"<папка>\\<ракурс>_parts.png") — PNG и JSON с цветами объектов.'
)

mcp = MCPServer(
    "zrenie",
    version=__version__,
    instructions=(
        "Зрение: показывает референсы с диска так, чтобы их было хорошо видно, для 3D-моделирования. "
        "Порядок работы: inspect(path) — обзор, размеры и кадры, если это лист из нескольких видов; "
        "split_views(path) — каждый кадр листа крупно; view(path, region, mode) — любая область крупно, "
        "режимы contrast / edges / tones помогают читать форму; sheet(paths) — несколько картинок на одном листе. "
        "Набор видов: silhouette(view) — объект без фона и его размеры; align_views(views, labels) — все виды одной "
        "высоты на общей линии низа, с направляющими и несостыковками между видами; export_views — PNG без фона "
        "и views.json для подложек в Blender / C4D. Вид задаётся строкой: «путь», «путь|N» (кадр N листа из inspect) "
        "или «путь|x1,y1,x2,y2». Точные замеры: grid(view) — сетка в пикселях оригинала или в процентах объекта; "
        "mark(view, points, lines) — нарисовать свои точки и линии, получить длины, углы, доли объекта и привязку "
        "к краю (snap) — так проверяй себя вместо оценки на глаз; contour(view) — контур точками и ось симметрии "
        "в координатах сцены (для кривых и профилей в Blender); palette(view) — цвета объекта для материалов. "
        "Проверка модели: отрендерь тот же ракурс (лучше ортографической камерой, фон прозрачный) и вызови "
        "compare(reference, render) — совпадение силуэтов в % и где модель шире, уже, чего не хватает, что лишнее; "
        "compare_views(references, renders) — все ракурсы сразу. Правь модель и сравнивай снова. "
        "Детали: parts(view, parts=[{name, points, box}]) — выделить детали по своим точкам (SAM 2.1), их размеры, "
        "место, наклон, цвет; потом вид «путь|имя» (или «путь|N|имя») работает с одной деталью во всех инструментах; "
        "compare_parts(reference, render) — модель по деталям против референса; рендер — карта деталей: "
        f"{BLENDER_HINT} "
        "Все координаты — пиксели оригинала. На картинках inspect, split_views, view, mark и parts по верхнему и левому "
        "краю — линейки с подписями в пикселях оригинала: координаты для region, points, lines и box читай по ним. "
        "Обзорные картинки (inspect, silhouette, contour, align_views, compare) уменьшены — мелкое смотри через view. "
        "Пути к файлам — только абсолютные."
    ),
)
RULERS_TEXT = "Линейки сверху (x) и слева (y) — в пикселях оригинала."


@lru_cache(maxsize=16)
def _cached_image(path: str, mtime_ns: int, size: int) -> PILImage.Image:
    return imaging.open_image(Path(path))


@lru_cache(maxsize=16)
def _cached_layout(path: str, mtime_ns: int, size: int) -> layout.Layout:
    return layout.find_views(_cached_image(path, mtime_ns, size))


@lru_cache(maxsize=32)
def _cached_matte(path: str, mtime_ns: int, size: int, box: tuple[int, int, int, int], method: str) -> matting.Matte:
    return matting.matte(_cached_image(path, mtime_ns, size).crop(box), method)


def _open(path: str) -> tuple[PILImage.Image, Path, tuple[str, int, int]]:
    """Картинка из кеша (сбрасывается, если файл изменился). Кешированную картинку не менять."""
    try:
        p = imaging.resolve_path(path)
        stat = p.stat()
        key = (str(p), stat.st_mtime_ns, stat.st_size)
        return _cached_image(*key), p, key
    except imaging.ImageLoadError as e:
        raise ToolError(str(e)) from None


def _check_max_edge(max_edge: int, upper: int = imaging.HIRES_MAX_EDGE) -> None:
    if not imaging.MIN_EDGE <= max_edge <= upper:
        raise ToolError(f"max_edge должен быть от {imaging.MIN_EDGE} до {upper}, получено {max_edge}")


def _check_mode(mode: str) -> None:
    if mode not in modes.MODES:
        raise ToolError(f"Режим должен быть одним из: {', '.join(modes.MODES)}; получено «{mode}»")


def _border_colour(img: PILImage.Image) -> str:
    a = np.asarray(imaging.to_viewable(img)[0].convert("RGB"))
    border = np.concatenate([a[:2].reshape(-1, 3), a[-2:].reshape(-1, 3), a[:, :2].reshape(-1, 3), a[:, -2:].reshape(-1, 3)])
    r, g, b = np.median(border, axis=0).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


def _box_text(box: layout.Box) -> str:
    return f"[{box.x1}, {box.y1}, {box.x2}, {box.y2}] — {box.w}×{box.h} px"


@mcp.tool(structured_output=False)
def list_media(folder: str, recursive: bool = False) -> str:
    """Список картинок и видео в папке: имя, размер в пикселях, вес файла.

    folder: абсолютный путь к папке.
    recursive: искать и во вложенных папках.
    """
    root = Path(folder).expanduser()
    if not root.is_absolute():
        raise ToolError(f"Нужен абсолютный путь, получено: {folder}")
    if not root.is_dir():
        raise ToolError(f"Папка не найдена: {root}")

    files = sorted(root.rglob("*") if recursive else root.iterdir(), key=lambda f: str(f).lower())
    images = [f for f in files if f.is_file() and f.suffix.lower() in imaging.IMAGE_EXTS]
    videos = [f for f in files if f.is_file() and f.suffix.lower() in imaging.VIDEO_EXTS]
    lines = [f"Папка: {root}", f"Картинок: {len(images)}, видео: {len(videos)}"]
    for f in images[:MAX_LIST]:
        name = f.relative_to(root)
        try:
            with PILImage.open(f) as img:
                dims = f"{img.width}×{img.height} px"
        except Exception:
            dims = "не открывается"
        lines.append(f"  {name} — {dims}, {imaging.file_size_text(f.stat().st_size)}")
    for f in videos[: max(0, MAX_LIST - len(images))]:
        lines.append(f"  [видео] {f.relative_to(root)} — {imaging.file_size_text(f.stat().st_size)}")
    if len(images) + len(videos) > MAX_LIST:
        lines.append(f"  … показаны первые {MAX_LIST}")
    return "\n".join(lines)


@mcp.tool(structured_output=False)
def inspect(path: str) -> list[Image | str]:
    """Обзор референса: картинка целиком, размеры, формат и кадры, если это лист из нескольких видов.

    Начинай работу с референсом с этого инструмента. Если найдено несколько кадров, они обведены
    пурпурными рамками с номерами, а их координаты перечислены в тексте.

    path: абсолютный путь к картинке.
    """
    img, p, key = _open(path)
    found = _cached_layout(*key)
    shown, flattened = imaging.render(img, max_tokens=imaging.OVERVIEW_TOKENS)
    scale = img.width / shown.width
    shown = measure.draw_rulers(shown, layout.Box(0, 0, img.width, img.height))
    if len(found.views) > 1:
        shown = draw.draw_boxes(shown, found.views, scale)
    data, fmt = imaging.encode(shown)

    lines = [
        f"Файл: {p}",
        f"Формат: {img.format or p.suffix.lstrip('.').upper()}, {img.width}×{img.height} px, "
        f"{imaging.file_size_text(p.stat().st_size)}" + (", есть прозрачность (залита серым)" if flattened else ""),
        f"Показано: {shown.width}×{shown.height} px"
        + (" (1:1)" if scale == 1 else f", координаты в оригинале = показанные × {scale:.4f}"),
    ]
    if len(found.views) > 1:
        lines.append(f"Это лист: найдено кадров — {len(found.views)} (по чему разрезано: {found.method}). Рамки и номера — на картинке.")
        lines += [f"  Кадр {i}: {_box_text(b)}" for i, b in enumerate(found.views, 1)]
    else:
        lines.append(f"Один кадр. Изображение без пустых полей: {_box_text(found.content)}.")
    if found.dropped:
        lines.append(f"Отброшено мелких кусков (подписи, тени): {found.dropped}.")
    lines.append(f"Цвет фона по краям: {_border_colour(img)}.")
    lines.append(RULERS_TEXT)
    lines.append(
        "Дальше: split_views — все кадры крупно; view(path, region=[x1, y1, x2, y2]) — любая область крупно; "
        "mode='contrast' | 'edges' | 'tones' — рельеф, контуры, свет и тень."
    )
    return [Image(data=data, format=fmt), "\n".join(lines)]


@mcp.tool(structured_output=False)
def split_views(path: str, max_edge: int = SPLIT_MAX_EDGE) -> list[Image | str]:
    """Разрезает лист (коллаж, turnaround) на отдельные кадры и показывает каждый крупно.

    Кадры ищутся автоматически: по однотонным промежуткам между ними или по швам, если кадры склеены встык.
    Если кадр один — показывает объект без пустых полей. Мелкие кадры увеличиваются.

    path: абсолютный путь к картинке.
    max_edge: потолок длинной стороны каждого кадра в px, от 200 до 2000 (по умолчанию 1024).
    """
    _check_max_edge(max_edge, imaging.DEFAULT_MAX_EDGE)
    img, p, key = _open(path)
    found = _cached_layout(*key)
    views = found.views if len(found.views) > 1 else [found.content]
    shown_views = views[:MAX_SPLIT_SHOWN]

    out: list[Image | str] = []
    for i, box in enumerate(shown_views, 1):
        prepared = imaging.prepare(img.crop(box.as_list()), max_edge, rulers=box)
        s = box.w / prepared.size[0]
        out.append(
            f"Кадр {i} из {len(views)}: область оригинала {_box_text(box)}; показан {prepared.size[0]}×{prepared.size[1]} px; "
            f"x = {box.x1} + x_показ × {s:.4f}, y = {box.y1} + y_показ × {s:.4f}"
        )
        out.append(Image(data=prepared.data, format=prepared.fmt))

    summary = [f"Файл: {p} ({img.width}×{img.height} px)"]
    if len(found.views) > 1:
        summary.append(f"Кадров: {len(views)} (по чему разрезано: {found.method}).")
    else:
        summary.append("Это не лист — один кадр, показан без пустых полей.")
    if len(views) > len(shown_views):
        summary.append(f"Показаны первые {len(shown_views)}; остальные смотри через view(path, region=...) по координатам из inspect.")
    summary.append(RULERS_TEXT)
    summary.append("Детали: view(path, region=[x1, y1, x2, y2], mode=...) в координатах оригинала.")
    out.append("\n".join(summary))
    return out


@mcp.tool(structured_output=False)
def view(
    path: str,
    region: list[float] | None = None,
    mode: str = "normal",
    max_edge: int = imaging.DEFAULT_MAX_EDGE,
) -> list[Image | str]:
    """Показать картинку или её область крупно, в максимальном качестве, которое Claude примет без пережатия.

    path: абсолютный путь к картинке (png, jpg, webp, bmp, gif, tiff).
    region: [x1, y1, x2, y2] в пикселях оригинала — показать только эту область. Мелкая область увеличивается.
    mode: "normal" — как есть; "contrast" — локальный контраст (фаски, швы, рельеф в тенях и бликах);
        "edges" — чёрные контуры на бледной подложке (силуэт и линии формы); "tones" — 5 уровней яркости
        (плоскости света и тени — читать объём).
    max_edge: потолок длинной стороны в px, от 200 до 2576. По умолчанию 2000 — так картинка
        не упрётся в лимит, когда в разговоре больше 20 картинок.

    Возвращает картинку с линейками по краям (подписи — пиксели оригинала) и текст с формулой перевода координат.
    """
    _check_max_edge(max_edge)
    _check_mode(mode)
    img, p, _ = _open(path)

    box = None
    if region is not None:
        if len(region) != 4:
            raise ToolError(f"region — это [x1, y1, x2, y2], получено {region}")
        x1, y1, x2, y2 = (round(v) for v in region)
        requested = [*sorted((x1, x2)), *sorted((y1, y2))]
        requested = [requested[0], requested[2], requested[1], requested[3]]
        box = [max(0, requested[0]), max(0, requested[1]), min(img.width, requested[2]), min(img.height, requested[3])]
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            raise ToolError(f"Область {region} вне картинки или слишком мала (картинка {img.width}×{img.height} px)")
        source = img.crop(box)
    else:
        source = img

    shown_box = layout.Box(*box) if box is not None else layout.Box(0, 0, img.width, img.height)
    prepared = imaging.prepare(source, max_edge, mode, rulers=shown_box)
    text = imaging.describe(p, img.size, prepared, box, mode)
    if box is not None and box != requested:
        text += f"\nОбласть подрезана по краям картинки: запрошено {requested}, показано {box}."
    text += f"\n{RULERS_TEXT}"
    # Картинка перед текстом — так Claude понимает изображения лучше.
    return [Image(data=prepared.data, format=prepared.fmt), text]


@mcp.tool(structured_output=False)
def sheet(paths: list[str], labels: list[str] | None = None, mode: str = "normal") -> list[Image | str]:
    """Несколько картинок на одном листе с подписями — сравнить виды рядом одной картинкой.

    Экономит лимит: в разговоре с больше чем 20 картинками каждая должна быть не больше 2000 px.

    paths: абсолютные пути, от 1 до 12 картинок.
    labels: подписи к картинкам (столько же, сколько путей). По умолчанию — номер и имя файла.
    mode: как в view — "normal", "contrast", "edges", "tones".
    """
    _check_mode(mode)
    if not 1 <= len(paths) <= MAX_SHEET:
        raise ToolError(f"Нужно от 1 до {MAX_SHEET} путей, получено {len(paths)}")
    if labels is not None and len(labels) != len(paths):
        raise ToolError(f"Подписей {len(labels)}, а картинок {len(paths)} — должно быть поровну")

    loaded = [_open(path) for path in paths]
    names = labels or [f"{i}: {p.name}" for i, (_, p, _) in enumerate(loaded, 1)]
    images = [modes.apply(imaging.to_viewable(img)[0], mode) for img, _, _ in loaded]
    composed, placements = draw.compose_sheet(images, names)
    shown, _ = imaging.render(composed)
    data, fmt = imaging.encode(shown, force_png=mode in modes.LINE_MODES)

    f = composed.width / shown.width
    lines = [f"Лист {shown.width}×{shown.height} px, режим «{modes.MODE_NAMES[mode]}». Где какая картинка:"]
    for name, (img, p, _), place in zip(names, loaded, placements):
        b = place.box
        x1, y1, x2, y2 = (round(v / f) for v in (b.x1, b.y1, b.x2, b.y2))
        s = place.scale * f
        lines.append(
            f"  {name} — на листе [{x1}, {y1}, {x2}, {y2}]; оригинал {img.width}×{img.height} px ({p}); "
            f"x = (x_лист − {x1}) × {s:.4f}, y = (y_лист − {y1}) × {s:.4f}"
        )
    return [Image(data=data, format=fmt), "\n".join(lines)]


def _source(spec: str, require_single: bool = True) -> vs.ViewSource:
    try:
        return vs.parse_spec(spec, _open, lambda key: _cached_layout(*key), require_single)
    except vs.ViewError as e:
        raise ToolError(str(e)) from None


def _check_method(method: str) -> None:
    if method not in matting.METHODS:
        raise ToolError(f"method должен быть одним из: {', '.join(matting.METHODS)}; получено «{method}»")


def _matte(source: vs.ViewSource, method: str) -> matting.Matte:
    if source.part:
        return _part_matte(source)
    stat = source.path.stat()
    try:
        return _cached_matte(str(source.path), stat.st_mtime_ns, stat.st_size, tuple(source.box.as_list()), method)
    except (ValueError, RuntimeError) as e:
        raise ToolError(str(e)) from None


def _in_original(source: vs.ViewSource, box: layout.Box) -> layout.Box:
    return layout.Box(source.box.x1 + box.x1, source.box.y1 + box.y1, source.box.x1 + box.x2, source.box.y1 + box.y2)


def _aligned(specs: list[str], labels: list[str] | None, method: str) -> list[vs.AlignedView]:
    _check_method(method)
    if not 1 <= len(specs) <= MAX_ALIGN:
        raise ToolError(f"Нужно от 1 до {MAX_ALIGN} видов, получено {len(specs)}")
    if labels is not None and len(labels) != len(specs):
        raise ToolError(f"Подписей {len(labels)}, а видов {len(specs)} — должно быть поровну")
    result = []
    for i, spec in enumerate(specs):
        source = _source(spec)
        matte = _matte(source, method)
        shape = vs.measure(matte)
        if shape is None:
            raise ToolError(f"На виде «{source.name}» объект не найден ({matte.method}). {matte.note}")
        label = labels[i] if labels else None
        title = vs.label_text(label, f"{i + 1}: {source.name}")
        result.append(vs.AlignedView(source, matte, shape, vs.normalize_label(label), title, 1.0))
    return result


@mcp.tool(structured_output=False)
def silhouette(view: str, method: str = "auto") -> list[Image | str]:
    """Отделить объект от фона и измерить силуэт: рамка, пропорции, части, отверстия, ключевые высоты.

    Показывает объект на бледном фоне с пурпурным контуром маски — проверь глазами, всё ли вошло.

    view: «путь», «путь|N» (кадр N листа из inspect), «путь|0» (вся картинка) или «путь|x1,y1,x2,y2» (область оригинала).
    method: "auto" — нейросеть BiRefNet (тени, отражения, сложный фон); "simple" — по цвету фона, только
        для однотонного фона; "ai" — только нейросеть.
    """
    _check_method(method)
    source = _source(view, require_single=False)
    matte = _matte(source, method)
    shape = vs.measure(matte)
    shown, _ = imaging.render(vs.silhouette_preview(source, matte, shape), max_tokens=imaging.OVERVIEW_TOKENS)
    data, fmt = imaging.encode(shown)
    s = source.box.w / shown.width

    lines = [f"Вид: {source.name}; область оригинала {_box_text(source.box)}", f"Способ: {matte.method}."]
    if matte.note:
        lines.append(matte.note)
    if source.sheet_views > 1 and vs.SPEC_SEPARATOR not in view:
        lines.append(f"Это лист из {source.sheet_views} кадров — силуэт по всему листу. Для одного вида: «путь|N».")
    if shape is None:
        lines.append("Объект не найден.")
    else:
        b = _in_original(source, shape.bbox)
        cx, cy = source.box.x1 + shape.centroid[0], source.box.y1 + shape.centroid[1]
        lines += [
            f"Объект: рамка в оригинале {_box_text(b)}; ширина / высота = {shape.aspect:.3f} (высота / ширина = {1 / shape.aspect:.3f}).",
            f"Площадь: {shape.area / (source.box.w * source.box.h) * 100:.1f} % области, заполняет рамку на "
            f"{shape.area / (b.w * b.h) * 100:.0f} %; отдельных частей: {shape.parts}; сквозных отверстий: {shape.holes}.",
            f"Центр масс в оригинале: ({cx:.0f}, {cy:.0f}).",
        ]
        if shape.key_heights:
            lines.append(
                "Ключевые высоты (где резко меняется ширина силуэта), от низа объекта: "
                + ", ".join(f"{k * 100:.0f} %" for k in shape.key_heights)
                + "."
            )
    lines.append(
        f"Показано {shown.width}×{shown.height} px; x = {source.box.x1} + x_показ × {s:.4f}, y = {source.box.y1} + y_показ × {s:.4f}."
    )
    return [Image(data=data, format=fmt), "\n".join(lines)]


@mcp.tool(structured_output=False)
def align_views(views: list[str], labels: list[str] | None = None, method: str = "auto") -> list[Image | str]:
    """Выровнять виды как на чертеже: фон убран, высота объекта одинаковая, низ на одной линии.

    Пурпурные пунктирные линии — низ (0 %), верх (100 %) и ключевые высоты первого вида (где резко меняется
    ширина силуэта): по ним видно, совпадают ли детали между видами. Вид сверху/снизу подгоняется по ширине
    вида спереди/сзади. В тексте — размеры, пропорции и несостыковки между видами.

    views: от 1 до 8 видов: «путь», «путь|N» (кадр N листа из inspect), «путь|0» (вся картинка) или «путь|x1,y1,x2,y2».
    labels: что это за виды — front/спереди, back/сзади, left/слева, right/справа, side/сбоку, top/сверху,
        bottom/снизу, 3-4. С подписями сервер проверяет пары (спереди ↔ сзади, слева ↔ справа, глубина сбоку ↔ сверху).
    method: способ убрать фон, как в silhouette.
    """
    aligned = _aligned(views, labels, method)
    canvas, guides = vs.compose_aligned(aligned)
    shown, _ = imaging.render(canvas, max_tokens=imaging.OVERVIEW_TOKENS)
    data, fmt = imaging.encode(shown)
    f = canvas.width / shown.width

    reference = next((v for v in aligned if v.label not in vs.PLAN_VIEWS), None)
    header = f"Лист {shown.width}×{shown.height} px. Объекты одной высоты, низ на общей линии. Пурпурные линии: низ 0 %, верх 100 %"
    if reference and guides:
        header += f" и ключевые высоты вида «{reference.title}»: " + ", ".join(f"{g * 100:.0f} %" for g in guides)
    lines = [header + "."]
    for v in aligned:
        b = _in_original(v.source, v.shape.bbox)
        p = v.placed
        k = f / v.scale
        lines.append(
            f"«{v.title}» — {v.source.name}; объект в оригинале {_box_text(b)}; ширина / высота = {v.shape.aspect:.3f}; "
            f"на листе [{p.x1 / f:.0f}, {p.y1 / f:.0f}, {p.x2 / f:.0f}, {p.y2 / f:.0f}]; "
            f"x_ориг = {b.x1} + (x_лист − {p.x1 / f:.0f}) × {k:.4f}, y_ориг = {b.y1} + (y_лист − {p.y1 / f:.0f}) × {k:.4f}. "
            f"Фон: {v.matte.method}." + (f" {v.matte.note}" if v.matte.note else "")
        )
    problems = vs.consistency_report(aligned)
    lines.append("Несостыковки между видами:" if problems else "Явных несостыковок по силуэтам не найдено.")
    lines += [f"  — {p}" for p in problems]
    if labels is None and len(aligned) > 1:
        lines.append("Подписи видов не заданы — пары (спереди ↔ сзади и т. д.) не проверены. Передай labels.")
    return [Image(data=data, format=fmt), "\n".join(lines)]


@mcp.tool(structured_output=False)
def export_views(
    views: list[str],
    folder: str,
    labels: list[str] | None = None,
    real_height: float | None = None,
    method: str = "auto",
) -> str:
    """Сохранить выровненные виды в PNG с прозрачным фоном + views.json — для подложек в Blender / C4D.

    У всех видов одинаковая высота объекта в пикселях (вид сверху/снизу — по ширине вида спереди/сзади).
    В views.json для каждого вида: размер картинки, где на ней объект, низ-центр объекта; если задана
    real_height — размер картинки в единицах сцены и смещение её центра от низа-центра объекта.

    views: виды, как в align_views. folder: абсолютный путь к папке (создаётся, файлы перезаписываются).
    labels: подписи видов (front, back, left, right, side, top, bottom, 3-4) — попадут в имена файлов.
    real_height: настоящая высота объекта в единицах сцены (например 0.2 для 20 см в метрах).
    method: способ убрать фон, как в silhouette.
    """
    out = Path(folder).expanduser()
    if not out.is_absolute():
        raise ToolError(f"Нужен абсолютный путь к папке, получено: {folder}")
    if real_height is not None and real_height <= 0:
        raise ToolError("real_height должен быть больше нуля")
    aligned = _aligned(views, labels, method)
    try:
        files, meta_path, meta = vs.export(aligned, out, real_height)
    except OSError as e:
        raise ToolError(f"Не удалось записать в {out}: {e}") from None

    lines = [f"Сохранено в {out}: {len(files)} PNG и {meta_path.name}. Высота объекта на всех видах: {meta['object_height_px']} px."]
    for record in meta["views"]:
        line = (
            f"  {record['file']} — «{record['title']}», {record['image_px'][0]}×{record['image_px'][1]} px, "
            f"объект {record['object_box_px']}, низ-центр объекта {record['object_bottom_center_px']}"
        )
        if real_height:
            line += (
                f"; картинка {record['image_size_units'][0]}×{record['image_size_units'][1]} ед., центр картинки "
                f"от низа-центра объекта {record['image_center_from_object_bottom_center_units']} (x вправо, вверх +)"
            )
        lines.append(line)
    if real_height:
        lines.append(f"Масштаб: {meta['units_per_px']:.6f} ед. на пиксель. Низ объекта — на 0, центр — на оси.")
    else:
        lines.append("Чтобы получить размеры в единицах сцены, передай real_height — настоящую высоту объекта.")
    return "\n".join(lines)


GRID_UNITS = ("px", "percent")
SNAP_MODES = ("none", "silhouette", "edges")


def _shape(source: vs.ViewSource, method: str) -> tuple[matting.Matte, vs.Shape]:
    matte = _matte(source, method)
    shape = vs.measure(matte)
    if shape is None:
        raise ToolError(f"На виде «{source.name}» объект не найден ({matte.method}). {matte.note}")
    return matte, shape


@mcp.tool(structured_output=False)
def grid(view: str, step: int | None = None, units: str = "px", mode: str = "normal", method: str = "auto") -> list[Image | str]:
    """Сетка координат поверх вида — снимать размеры и положение деталей по подписанным линиям.

    view: «путь», «путь|N» (кадр листа), «путь|0» или «путь|x1,y1,x2,y2» — сетку удобно класть на область крупно.
    step: шаг сетки в пикселях оригинала (units="px"); по умолчанию — круглый, ~10 линий по длинной стороне.
    units: "px" — подписи в пикселях оригинала; "percent" — сетка по рамке объекта: 0–100 % ширины слева
        направо и 0–100 % высоты снизу вверх, линии через 5 % (толще — 0, 50, 100 %).
    mode: подложка как в view — "normal", "contrast", "edges", "tones".
    method: способ найти объект для units="percent", как в silhouette.
    """
    _check_mode(mode)
    if units not in GRID_UNITS:
        raise ToolError(f"units должен быть одним из: {', '.join(GRID_UNITS)}")
    if step is not None and step < 1:
        raise ToolError("step должен быть не меньше 1 px")
    source = _source(view, require_single=False)
    shown, _ = imaging.render(source.image, imaging.DEFAULT_MAX_EDGE, mode)
    region = source.box
    s = region.w / shown.width
    lines = [f"Вид: {source.name}; область оригинала {_box_text(region)}; показано {shown.width}×{shown.height} px."]
    if units == "px":
        step = step or measure.nice_step(max(region.w, region.h))
        picture = measure.draw_pixel_grid(shown, region, step)
        lines.append(
            f"Сетка в пикселях оригинала: подписанные линии через {step} px, тонкие — через {step / 5:g} px (если влезают). "
            "Подписи сверху — x, слева — y."
        )
    else:
        _, shape = _shape(source, method)
        obj = _in_original(source, shape.bbox)
        picture = measure.draw_percent_grid(shown, region, obj)
        lines.append(
            f"Сетка в процентах рамки объекта {_box_text(obj)}: x — от левого края (0 %) к правому (100 %), "
            "высота — от низа (0 %) к верху (100 %); линии через 5 %, подписи через 10 %."
        )
        lines.append(f"1 % ширины = {obj.w / 100:.2f} px, 1 % высоты = {obj.h / 100:.2f} px.")
    lines.append(f"Координаты в оригинале: x = {region.x1} + x_показ × {s:.4f}, y = {region.y1} + y_показ × {s:.4f}.")
    data, fmt = imaging.encode(picture, force_png=True)
    return [Image(data=data, format=fmt), "\n".join(lines)]


def _numbers(items: list[list[float]] | None, size: int, what: str) -> list[tuple[float, ...]]:
    result = []
    for item in items or []:
        if len(item) != size:
            raise ToolError(f"{what}: каждый элемент — {size} числа, получено {item}")
        result.append(tuple(float(v) for v in item))
    return result


@mcp.tool(structured_output=False)
def mark(
    view: str,
    points: list[list[float]] | None = None,
    lines: list[list[float]] | None = None,
    labels: list[str] | None = None,
    snap: str = "none",
    method: str = "auto",
) -> list[Image | str]:
    """Нарисовать свои точки и линии на виде и получить точные числа — проверка себя вместо оценки на глаз.

    Показывает область вокруг отметок крупно. Для каждой линии — длина в px, в % высоты и ширины объекта,
    угол; для каждой точки — положение в % рамки объекта. Если отметка легла не туда — поправь и вызови снова.

    view: «путь», «путь|N», «путь|0» или «путь|x1,y1,x2,y2». Координаты точек и линий — пиксели оригинала.
    points: [[x, y], ...]. lines: [[x1, y1, x2, y2], ...]. labels: подписи точек (по умолчанию P1, P2, …).
    snap: "none"; "silhouette" — притянуть точки и концы линий к ближайшему краю силуэта (до 12 px);
        "edges" — к ближайшему резкому перепаду внутри объекта (край детали, шов, фаска).
    method: способ найти объект (для процентов и snap="silhouette"), как в silhouette.
    """
    if snap not in SNAP_MODES:
        raise ToolError(f"snap должен быть одним из: {', '.join(SNAP_MODES)}")
    pts = _numbers(points, 2, "points")
    segs = _numbers(lines, 4, "lines")
    if not pts and not segs:
        raise ToolError("Нужна хотя бы одна точка (points) или линия (lines)")
    if labels is not None and len(labels) != len(pts):
        raise ToolError(f"Подписей {len(labels)}, а точек {len(pts)} — должно быть поровну")
    source = _source(view, require_single=False)
    matte, shape = _shape(source, method)
    obj = _in_original(source, shape.bbox)
    origin = (source.box.x1, source.box.y1)

    text = [f"Вид: {source.name}; объект в оригинале {_box_text(obj)}."]
    if snap != "none":
        what = "краю силуэта" if snap == "silhouette" else "ближайшему перепаду"
        snapped = measure.snap_points(list(pts), source.image, origin, matte.mask, snap)
        pts = [r.snapped or r.point for r in snapped]
        moved, total, notes = sum(r.snapped is not None for r in snapped), len(snapped), []
        new_segs = []
        for i, seg in enumerate(segs, 1):
            line, shifts = measure.snap_line(seg, source.image, origin, matte.mask, snap)
            new_segs.append(line)
            total += 2
            moved += sum(v is not None for v in shifts)
            notes.append(
                f"L{i}: концы сдвинуты вдоль линии на "
                + ", ".join("—" if v is None else f"{v:+.1f} px" for v in shifts)
            )
        segs = new_segs
        text.append(
            f"Привязка к {what}: сдвинуто {moved} из {total} (не дальше {measure.SNAP_RADIUS} px; концы линий — только вдоль линии)."
        )
        text += notes

    names = labels or [f"P{i + 1}" for i in range(len(pts))]
    for name, (x, y) in zip(names, pts):
        rx = (x - obj.x1) / obj.w * 100
        rz = (obj.y2 - y) / obj.h * 100
        text.append(f"{name}: ({x:.1f}, {y:.1f}) — {rx:.1f} % ширины объекта от левого края, {rz:.1f} % высоты от низа.")
    for i, (x1, y1, x2, y2) in enumerate(segs, 1):
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(-dy, dx))
        text.append(
            f"L{i}: ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f}); длина {length:.1f} px = {length / obj.h * 100:.1f} % высоты "
            f"= {length / obj.w * 100:.1f} % ширины объекта; угол {angle:.1f}° от горизонтали; Δx {dx:+.1f}, Δy {dy:+.1f}."
        )

    region = measure.marks_region(list(pts), list(segs), source.box)
    crop = source.image.crop((region.x1 - origin[0], region.y1 - origin[1], region.x2 - origin[0], region.y2 - origin[1]))
    shown, _ = imaging.render(crop)
    picture = measure.draw_marks(measure.draw_rulers(shown, region), region, list(pts), names, list(segs))
    s = region.w / shown.width
    text.append(f"Показана область {_box_text(region)}; x = {region.x1} + x_показ × {s:.4f}, y = {region.y1} + y_показ × {s:.4f}.")
    text.append(RULERS_TEXT)
    data, fmt = imaging.encode(picture)
    return [Image(data=data, format=fmt), "\n".join(text)]


@mcp.tool(structured_output=False)
def contour(
    view: str,
    detail: float = measure.CONTOUR_DETAIL,
    half: bool = False,
    real_height: float | None = None,
    save_to: str | None = None,
    method: str = "auto",
) -> list[Image | str]:
    """Контур силуэта точками и вертикальная ось симметрии — в координатах сцены, готово для кривых в Blender.

    Координаты: начало на оси симметрии у низа объекта, x вправо, z вверх. Единица — высота объекта (1.0),
    или единицы сцены, если задана real_height. Внешний контур и отверстия — отдельными списками.

    view: «путь», «путь|N», «путь|0» или «путь|x1,y1,x2,y2».
    detail: точность упрощения — доля диагонали объекта (0.003 по умолчанию; меньше — больше точек).
    half: только правая половина внешнего контура от верха до низа — профиль для тела вращения (Screw).
    real_height: настоящая высота объекта в единицах сцены.
    save_to: абсолютный путь к .json — сохранить контуры, чтобы прочитать их кодом в Blender.
    method: способ найти объект, как в silhouette.
    """
    if not 0.0005 <= detail <= 0.05:
        raise ToolError("detail должен быть от 0.0005 до 0.05")
    if real_height is not None and real_height <= 0:
        raise ToolError("real_height должен быть больше нуля")
    source = _source(view, require_single=False)
    matte, shape = _shape(source, method)
    bbox = shape.bbox
    contours = measure.find_contours(matte.mask, bbox, detail)
    sym = measure.find_symmetry(matte.mask, bbox)
    symmetric = sym.score >= 0.85
    axis = sym.axis_x if symmetric else (bbox.x1 + bbox.x2) / 2
    units_per_px = (real_height or 1.0) / bbox.h
    unit = "ед. сцены" if real_height else "высоты объекта"

    shown, _ = imaging.render(vs.silhouette_preview(source, matte, None), max_tokens=imaging.OVERVIEW_TOKENS)
    picture = measure.draw_contours(shown, source.box.w / shown.width, contours, axis)
    data, fmt = imaging.encode(picture, force_png=True)

    obj = _in_original(source, bbox)
    if sym.score >= 0.95:
        verdict = "— симметричен."
    elif symmetric:
        verdict = "— почти симметричен."
    else:
        verdict = "— несимметричен, начало координат по центру рамки."
    text = [
        f"Вид: {source.name}; объект в оригинале {_box_text(obj)}; способ: {matte.method}.",
        f"Ось симметрии: x = {source.box.x1 + axis:.1f} px оригинала; силуэт совпадает с зеркальным на {sym.score * 100:.1f} % {verdict}",
        f"Координаты: (x, z), начало на оси у низа объекта, x вправо, z вверх, единица — {unit} "
        f"(1 px = {units_per_px:.6f}). Ширина объекта = {bbox.w * units_per_px:.4f}, высота = {bbox.h * units_per_px:.4f}.",
        "На картинке: пурпурный — внешний контур, голубой — отверстия, зелёный пунктир — ось.",
    ]
    record = {
        "view": view,
        "units": unit,
        "units_per_px": units_per_px,
        "axis_x_px": source.box.x1 + axis,
        "bottom_y_px": source.box.y1 + bbox.y2,
        "symmetry": round(sym.score, 4),
        "contours": [],
    }
    for i, c in enumerate(contours, 1):
        scene = measure.to_scene(c.points, axis, bbox.y2, units_per_px)
        record["contours"].append({"kind": c.kind, "points": [[round(x, 5), round(z, 5)] for x, z in scene]})
        shown_pts = scene[: measure.CONTOUR_MAX_POINTS]
        name = "внешний" if c.kind == "outer" else "отверстие"
        cut = f", показаны первые {len(shown_pts)}" if len(scene) > len(shown_pts) else ""
        text.append(f"Контур {i} ({name}, {len(scene)} точек{cut}):")
        text.append(" ".join(f"({x:.4f}, {z:.4f})" for x, z in shown_pts))
    if half:
        outer = next((c for c in contours if c.kind == "outer"), None)
        profile = measure.half_profile(outer.points, axis) if outer is not None else np.empty((0, 2))
        scene = measure.to_scene(profile, axis, bbox.y2, units_per_px) if len(profile) else np.empty((0, 2))
        record["half_profile"] = [[round(x, 5), round(z, 5)] for x, z in scene]
        text.append(f"Профиль (правая половина, сверху вниз, {len(scene)} точек):")
        text.append(" ".join(f"({x:.4f}, {z:.4f})" for x, z in scene))
        if not symmetric:
            text.append("Внимание: объект несимметричен — профиль для тела вращения будет неточным.")
    if save_to:
        out = Path(save_to).expanduser()
        if not out.is_absolute():
            raise ToolError(f"save_to — нужен абсолютный путь, получено: {save_to}")
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as e:
            raise ToolError(f"Не удалось записать {out}: {e}") from None
        text.append(f"Сохранено: {out}")
    return [Image(data=data, format=fmt), "\n".join(text)]


@mcp.tool(structured_output=False)
def palette(view: str, colors: int = 6, whole: bool = False, method: str = "auto") -> list[Image | str]:
    """Основные цвета объекта для материалов: HEX, sRGB 0..1 и линейные 0..1, доля площади.

    Цвета сняты с освещением: блики и тени попадают отдельными цветами. Базовый цвет материала — обычно
    самый частый средний тон, а не самый светлый или тёмный.

    view: «путь», «путь|N», «путь|0» или «путь|x1,y1,x2,y2» (например, только одна деталь).
    colors: сколько цветов, от 1 до 12.
    whole: True — считать по всей области вместе с фоном, без вырезки объекта.
    method: способ найти объект, как в silhouette.
    """
    if not 1 <= colors <= 12:
        raise ToolError("colors должен быть от 1 до 12")
    source = _source(view, require_single=False)
    matte = None if whole else _matte(source, method)
    swatches = measure.palette(source.image, None if matte is None else matte.mask, colors)
    thumb = vs.silhouette_preview(source, matte, None) if matte is not None else source.image
    shown, _ = imaging.render(measure.draw_palette(thumb, swatches))
    data, fmt = imaging.encode(shown, force_png=True)
    text = [f"Вид: {source.name}; цвета {'всей области' if whole else 'объекта без фона'}, от самого частого:"]
    for i, sw in enumerate(swatches, 1):
        text.append(f"  {i}. {sw.hex} — {sw.share * 100:.1f} %; sRGB 0..1 {sw.srgb01}; линейные 0..1 {sw.linear01}")
    text.append("Blender: HEX вводится в поле Hex (это sRGB), а числа в Base Color — линейные. Cinema 4D: HEX или sRGB 0..1.")
    return [Image(data=data, format=fmt), "\n".join(text)]


def _render_matte(source: vs.ViewSource, method: str) -> matting.Matte:
    """Силуэт рендера: по прозрачности, если фон прозрачный, иначе — как у референса."""
    return cmp.alpha_matte(source.image) or _matte(source, method)


def _compare_pair(reference: str, render: str, method: str) -> tuple[vs.ViewSource, vs.ViewSource, cmp.Comparison, matting.Matte, matting.Matte]:
    ref = _source(reference)
    model = _source(render, require_single=False)  # рендер — всегда один объект, даже если он из отдельных частей
    if ref.part or model.part:
        raise ToolError("compare сравнивает объект целиком. По деталям — compare_parts(reference, render) с картой деталей.")
    ref_matte = _matte(ref, method)
    model_matte = _render_matte(model, method)
    result = cmp.compare(ref.image, ref_matte, model.image, model_matte)
    if result is None:
        empty = ref.name if vs.measure(ref_matte) is None else model.name
        raise ToolError(f"На «{empty}» объект не найден — сравнивать нечего.")
    return ref, model, result, ref_matte, model_matte


@mcp.tool(structured_output=False)
def compare(reference: str, render: str, mode: str = "all", method: str = "auto") -> list[Image | str]:
    """Сравнить рендер модели с референсом того же ракурса: совпадение силуэтов в % и где именно расходятся.

    Силуэт рендера выравнивается по высоте и низу силуэта референса (масштаб модели в сцене не важен — важны
    пропорции). В тексте: совпадение (IoU — доля общей площади), где модель шире или уже по полосам высоты, чего
    не хватает и что лишнее, сдвиг ключевых высот. Правь модель и вызывай снова, пока совпадение не станет высоким.

    reference: вид референса — «путь», «путь|N» (кадр листа), «путь|0» или «путь|x1,y1,x2,y2».
    render: файл рендера того же ракурса (лучше ортографическая камера; прозрачный фон — силуэт точный).
    mode: "all" — референс, модель и разница рядом; "side" — только референс и модель; "diff" — только
        разница; "overlay" — наложение 50/50.
    method: способ найти объект на референсе (и на рендере без прозрачности), как в silhouette.
    """
    _check_method(method)
    if mode not in COMPARE_MODES:
        raise ToolError(f"mode должен быть одним из: {', '.join(COMPARE_MODES)}")
    ref, model, result, ref_matte, model_matte = _compare_pair(reference, render, method)
    picture = cmp.compose_rows([("", cmp.panels(result, mode))])
    shown, _ = imaging.render(picture, max_tokens=imaging.OVERVIEW_TOKENS)
    data, fmt = imaging.encode(shown, force_png=True)
    lines = [
        f"Совпадение силуэтов: {result.iou * 100:.1f} % (IoU).",
        f"Референс: {ref.name}, объект {_box_text(_in_original(ref, result.ref_box))}; фон: {ref_matte.method}.",
        f"Модель: {model.name}, силуэт: {model_matte.method}; выровнена по высоте и низу (масштаб ×{result.scale:.3f}).",
        cmp.LEGEND,
    ]
    lines += result.findings or ["Заметных расхождений по силуэту нет."]
    return [Image(data=data, format=fmt), "\n".join(lines)]


@mcp.tool(structured_output=False)
def compare_views(
    references: list[str], renders: list[str], labels: list[str] | None = None, method: str = "auto"
) -> list[Image | str]:
    """Сравнить сразу несколько ракурсов: по строке на ракурс — референс, модель, разница — и % совпадения каждого.

    references и renders — пары одного ракурса в одинаковом порядке (до 6 пар). Виды референса — как в compare.
    labels: названия ракурсов (front, side, top…). method: способ найти объект, как в silhouette.
    """
    _check_method(method)
    if not 1 <= len(references) <= MAX_COMPARE:
        raise ToolError(f"Нужно от 1 до {MAX_COMPARE} пар, получено {len(references)}")
    if len(renders) != len(references):
        raise ToolError(f"Референсов {len(references)}, а рендеров {len(renders)} — должно быть поровну")
    if labels is not None and len(labels) != len(references):
        raise ToolError(f"Подписей {len(labels)}, а пар {len(references)} — должно быть поровну")
    rows, lines, scores = [], [], []
    for i, (reference, render) in enumerate(zip(references, renders)):
        ref, model, result, _, model_matte = _compare_pair(reference, render, method)
        title = vs.label_text(labels[i] if labels else None, f"{i + 1}: {ref.name}")
        scores.append(result.iou)
        rows.append((f"{title} · совпадение {result.iou * 100:.1f} %", cmp.panels(result, "all")))
        lines.append(f"«{title}»: {result.iou * 100:.1f} % — {ref.name} ↔ {model.name} (силуэт модели: {model_matte.method}).")
        lines += [f"    {f}" for f in result.findings] or ["    Заметных расхождений нет."]
    picture = cmp.compose_rows(rows)
    shown, _ = imaging.render(picture, max_tokens=imaging.OVERVIEW_TOKENS)
    data, fmt = imaging.encode(shown, force_png=True)
    worst = min(range(len(scores)), key=lambda i: scores[i])
    header = [
        f"Среднее совпадение: {sum(scores) / len(scores) * 100:.1f} %; хуже всего — «{rows[worst][0].split(' · ')[0]}» ({scores[worst] * 100:.1f} %).",
        cmp.LEGEND,
    ]
    return [Image(data=data, format=fmt), "\n".join(header + lines)]


# ---------- детали ----------

# Выделенные детали: вид (файл, его версия, область) → имя детали → деталь. Живут до перезапуска сервера.
_parts: dict[tuple, dict[str, pt.Part]] = {}


def _view_key(source: vs.ViewSource) -> tuple:
    stat = source.path.stat()
    return (str(source.path), stat.st_mtime_ns, stat.st_size, tuple(source.box.as_list()))


def _part_matte(source: vs.ViewSource) -> matting.Matte:
    known = _parts.get(_view_key(source), {})
    part = known.get(source.part)
    if part is None:
        view_name = source.name.rsplit(", деталь", 1)[0]
        hint = f" На этом виде выделены: {', '.join(known)}." if known else ""
        raise ToolError(
            f"Детали «{source.part}» на виде {view_name} нет — сначала выдели её: parts(view, parts=[{{name, points / box}}]).{hint} "
            "Детали помнятся до перезапуска сервера и привязаны к той же области вида (тот же кадр «|N»)."
        )
    return matting.Matte(part.mask.astype(np.float32), f"деталь «{part.name}» (SAM 2.1)" if part.prompt else f"деталь «{part.name}»")


class PartPrompt(BaseModel):
    """Подсказка для одной детали: points и/или box в пикселях оригинала."""

    name: str = Field(description="Имя детали, например «чашка»; потом деталь берётся видом «путь|чашка».")
    points: list[list[float]] | None = Field(default=None, description="Точки на детали: [[x, y], ...].")
    exclude: list[list[float]] | None = Field(default=None, description="Точки, которые в деталь входить не должны: [[x, y], ...].")
    box: list[float] | None = Field(default=None, description="Рамка вокруг детали: [x1, y1, x2, y2].")
    pick: str = Field(
        default="auto",
        description="Только для одной точки без exclude и box: auto — самый уверенный вариант; small / medium / large — "
        "мельче или крупнее (часть детали, деталь, деталь с соседними).",
    )


def _prompt(p: PartPrompt, source: vs.ViewSource) -> pt.Prompt:
    name = p.name.strip()
    if not name or vs.SPEC_SEPARATOR in name or vs.is_frame_selector(name):
        raise ToolError(f"Имя детали «{p.name}» не подходит: не пустое, без «|» и запятых, не число.")
    ox, oy = source.box.x1, source.box.y1
    w, h = source.box.w, source.box.h

    def local(item: list[float], what: str) -> tuple[float, float]:
        if len(item) != 2:
            raise ToolError(f"«{name}»: {what} — это [x, y], получено {item}")
        x, y = item[0] - ox, item[1] - oy
        if not (0 <= x < w and 0 <= y < h):
            raise ToolError(f"«{name}»: точка {item} вне вида {_box_text(source.box)}")
        return x, y

    points = [local(q, "точка") for q in p.points or []]
    exclude = [local(q, "точка exclude") for q in p.exclude or []]
    box = None
    if p.box is not None:
        if len(p.box) != 4:
            raise ToolError(f"«{name}»: box — это [x1, y1, x2, y2], получено {p.box}")
        x1, x2 = sorted((p.box[0] - ox, p.box[2] - ox))
        y1, y2 = sorted((p.box[1] - oy, p.box[3] - oy))
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise ToolError(f"«{name}»: box {p.box} вне вида {_box_text(source.box)}")
        box = (x1, y1, x2, y2)
    if not points and box is None:
        raise ToolError(f"«{name}»: нужна хотя бы одна точка на детали (points) или рамка (box).")
    if p.pick not in pt.PICKS:
        raise ToolError(f"«{name}»: pick должен быть одним из: {', '.join(pt.PICKS)}")
    return pt.Prompt(name, points, exclude, box, p.pick)


@mcp.tool(structured_output=False)
def parts(view: str, parts: list[PartPrompt], method: str = "auto") -> list[Image | str]:
    """Выделить детали объекта по своим точкам и рамкам (нейросеть SAM 2.1) и измерить каждую.

    Посмотри на вид, поставь точку в середину детали (или несколько точек, или рамку box вокруг неё) — сервер
    вернёт маску детали цветом с номером и числа: размер и место в % рамки всего объекта, площадь, куски,
    отверстия, наклон длинной оси, цвет. Одна точка неоднозначна (деталь, её часть или весь объект) — если
    выделилось не то, добавь точки, рамку или точки exclude на соседних деталях и вызови снова с тем же именем.

    Выделенные детали запоминаются: вид «путь|имя» (для листа «путь|N|имя») дальше работает с одной деталью
    в silhouette, contour (half — профиль детали), palette, mark, grid (percent — сетка по детали), align_views
    (одна деталь на разных видах). compare_parts — сравнение модели с референсом по деталям.

    view: «путь», «путь|N» (кадр листа), «путь|0» или «путь|x1,y1,x2,y2».
    parts: до 12 деталей: [{"name": "чашка", "points": [[x, y]], "exclude": [[x, y]], "box": [x1, y1, x2, y2]}],
        координаты — пиксели оригинала. Надёжнее всего — 2–3 точки на детали и exclude на соседних деталях
        того же цвета; для одной точки есть pick: "small" | "medium" | "large".
    method: способ найти весь объект (для процентов), как в silhouette.
    """
    _check_method(method)
    if not 1 <= len(parts) <= MAX_PARTS:
        raise ToolError(f"Нужно от 1 до {MAX_PARTS} деталей, получено {len(parts)}")
    source = _source(view)
    if source.part:
        raise ToolError(f"view — вид без детали в конце, например «{view.rsplit(vs.SPEC_SEPARATOR, 1)[0]}».")
    prompts = [_prompt(p, source) for p in parts]
    names = [p.name for p in prompts]
    if len(set(names)) != len(names):
        raise ToolError("Имена деталей повторяются — у каждой детали своё имя.")
    if not matting.ai_available():
        raise ToolError("Нейросеть SAM 2.1 не установлена (нужны torch и transformers).")
    try:
        found = pt.segment(source.image, prompts)
    except OSError as e:
        raise ToolError(f"Не удалось загрузить SAM 2.1: {e}") from None
    remembered = _parts.setdefault(_view_key(source), {})
    remembered.update({p.name: p for p in found})
    # Номер и цвет детали — по её месту в памяти вида: одинаковые в parts и compare_parts, при повторных вызовах тоже.
    numbers = [list(remembered).index(p.name) + 1 for p in found]

    obj_matte = _matte(source, method)
    obj_shape = vs.measure(obj_matte)
    shown, _ = imaging.render(source.image)
    s = source.box.w / shown.width
    picture = measure.draw_rulers(pt.draw_parts(shown, found, s, numbers), source.box)
    data, fmt = imaging.encode(picture)

    origin = (source.box.x1, source.box.y1)
    lines = [f"Вид: {source.name}; область оригинала {_box_text(source.box)}. Детали — SAM 2.1 ({pt.device_name()})."]
    if obj_shape is None:
        lines.append(f"Весь объект не найден ({obj_matte.method}) — проценты не посчитаны.")
    for i, part in zip(numbers, found):
        colour = pt.COLOURS[(i - 1) % len(pt.COLOURS)][1]
        shape = pt.measure_part(part.mask, source.image)
        if shape is None:
            lines.append(f"{i}. «{part.name}» ({colour}) — пусто.")
            continue
        text = f"{i}. «{part.name}» ({colour}), уверенность {part.score:.2f}"
        if part.score < LOW_SCORE:
            text += " (низкая — проверь глазами)"
        if obj_shape is not None:
            text += f"; {pt.relative_text(shape, obj_shape.bbox, origin)}; площадь {shape.area / obj_shape.area * 100:.1f} % объекта"
        else:
            b = shape.bbox
            text += f"; рамка в оригинале [{origin[0] + b.x1}, {origin[1] + b.y1}, {origin[0] + b.x2}, {origin[1] + b.y2}] — {b.w}×{b.h} px"
        text += (
            f"; кусков {shape.pieces}, отверстий {shape.holes}; {pt.angle_text(shape.angle, shape.elongation)}; "
            f"цвет {shape.colour}."
        )
        if part.options:
            base = obj_shape.area if obj_shape is not None else shape.area
            unit = "объекта" if obj_shape is not None else "выбранной"
            chosen = shape.area
            text += " Варианты по одной точке (pick): " + ", ".join(
                f"{size} — {area / base * 100:.1f} % {unit} ({score:.2f})" + (" ← выбран" if area == chosen else "")
                for size, area, score in part.options
            ) + "."
        lines.append(text)
    problems = pt.warnings(found, obj_matte.mask if obj_shape is not None else None)
    if problems:
        lines.append("Проверь:")
        lines += [f"  — {p}" for p in problems]
    lines.append(
        f"Детали этого вида в памяти: {', '.join(remembered)}. Одна деталь — вид «{view}|имя» в silhouette, contour, palette, "
        "mark, grid, align_views. На картинке: белые точки — твои точки, крестики цвета детали — exclude, рамка — box."
    )
    lines.append(f"Показано {shown.width}×{shown.height} px; x = {source.box.x1} + x_показ × {s:.4f}, y = {source.box.y1} + y_показ × {s:.4f}.")
    lines.append(RULERS_TEXT)
    return [Image(data=data, format=fmt), "\n".join(lines)]


@mcp.tool(structured_output=False)
def compare_parts(
    reference: str, render: str, match: dict[str, list[str]] | None = None, method: str = "auto"
) -> list[Image | str]:
    """Сравнить модель с референсом по деталям: совпадение каждой детали в %, где она шире, уже, сдвинута, наклонена.

    Общий % силуэта не замечает ошибок в мелких деталях — здесь каждая деталь сравнивается отдельно. Модель
    выравнивается по всему объекту (высота и низ, как в compare), поэтому сдвиг детали тоже виден.

    reference: вид референса, на котором уже выделены детали через parts (тот же «путь» или «путь|N»).
    render: карта деталей того же ракурса — рендер, где каждый объект залит своим плоским цветом, фон прозрачный,
        рядом JSON с цветами объектов. Как сделать в Blender — в описании сервера (blender_kit.render_parts);
        render_parts возвращает, какие объекты видны с этой камеры.
    match: какие объекты модели — какая деталь референса: {"чашка": ["shell", "cushion_L"], "вилка": ["yoke"]}.
        По умолчанию — объекты с тем же именем, что и деталь.
    method: способ найти весь объект на референсе, как в silhouette.
    """
    _check_method(method)
    ref = _source(reference)
    if ref.part:
        raise ToolError("reference — вид без детали в конце: детали берутся все, что выделены через parts.")
    ref_parts = _parts.get(_view_key(ref), {})
    if not ref_parts:
        raise ToolError(f"На виде {ref.name} нет выделенных деталей — сначала parts(view, parts=[...]).")
    model = _source(render, require_single=False)
    try:
        colours = pt.read_id_map(model.path)
    except (ValueError, OSError) as e:
        raise ToolError(f"{e} {BLENDER_HINT}") from None
    model_matte = cmp.alpha_matte(model.image)
    if model_matte is None:
        raise ToolError("У карты деталей должен быть прозрачный фон (film_transparent).")

    if not match:  # пустой match — то же, что не задан
        lower = {n.lower(): n for n in colours}
        pairs = [(name, [lower[name.lower()]]) for name in ref_parts if name.lower() in lower]
        if not pairs:
            raise ToolError(
                f"Имена деталей ({', '.join(ref_parts)}) не совпадают с объектами модели ({', '.join(colours)}). "
                "Передай match: {\"деталь\": [\"объект\", ...]}."
            )
    else:
        pairs = []
        for name, objects in match.items():
            if name not in ref_parts:
                raise ToolError(f"Детали «{name}» нет на виде {ref.name}; выделены: {', '.join(ref_parts)}.")
            unknown = [o for o in objects if o not in colours]
            if unknown:
                raise ToolError(f"Объектов {', '.join(unknown)} нет в карте деталей; есть: {', '.join(colours)}.")
            pairs.append((name, list(objects)))

    ref_matte = _matte(ref, method)
    result = cmp.compare(ref.image, ref_matte, model.image, model_matte)
    if result is None:
        raise ToolError("На референсе или на рендере объект не найден — сравнивать нечего.")
    h, w = result.ref_mask.shape  # холст сравнения: вид референса с запасом, если модель шире кадра
    masks = pt.model_parts(model.image, colours)
    warped = {
        name: cv2.warpAffine(m.astype(np.float32), result.matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0) >= 0.5
        for name, m in masks.items()
    }

    obj = result.ref.bbox
    ref_masks = {name: result.to_canvas(ref_parts[name].mask) for name, _ in pairs}
    rows = []
    for name, objects in pairs:
        model_mask = np.zeros((h, w), bool)
        for o in objects:
            model_mask |= warped[o]
        rows.append((pt.compare_part(name, ref_masks[name], model_mask, obj), objects, model_mask))

    numbered = {n: list(ref_parts).index(n) for n, _ in pairs}  # цвет — как у детали в parts
    panels = pt.compare_panels(
        result.ref_rgb, [ref_masks[n] for n, _ in pairs], [m for _, _, m in rows], result.model_mask, result.shown_area(), [numbered[n] for n, _ in pairs]
    )
    shown, _ = imaging.render(cmp.compose_rows([("", panels)]), max_tokens=imaging.OVERVIEW_TOKENS)
    data, fmt = imaging.encode(shown, force_png=True)

    scores = [c.iou for c, _, _ in rows]
    lines = [
        f"Совпадение по деталям: среднее {sum(scores) / len(scores) * 100:.1f} %; весь силуэт — {result.iou * 100:.1f} %.",
        f"Референс: {ref.name}; модель: {model.name}, выровнена по всему объекту (масштаб ×{result.scale:.3f}).",
        "Детали от худшей к лучшей; размеры рамки детали — модель против референса (+ больше, − меньше):",
    ]
    for c, objects, _ in sorted(rows, key=lambda r: r[0].iou):
        colour = pt.COLOURS[numbered[c.name] % len(pt.COLOURS)][1]
        lines.append(f"  «{c.name}» ({colour}) ↔ {', '.join(objects)}: {c.iou * 100:.1f} % — " + ("; ".join(c.findings) or "совпадает") + ".")
    used = {o for _, objects in pairs for o in objects}
    visible = [n for n, m in masks.items() if m.any() and n not in used]
    if visible:
        lines.append(f"Объекты модели без пары (видны на рендере): {', '.join(visible)}.")
    unmatched = [n for n in ref_parts if n not in numbered]
    if unmatched:
        lines.append(f"Детали референса без пары: {', '.join(unmatched)}.")
    return [Image(data=data, format=fmt), "\n".join(lines)]