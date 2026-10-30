"""Подготовка картинок для Claude: точный размер, увеличение мелкого, формат, описание для координат."""

import io
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from . import measure, modes
from .layout import Box

# Лимиты Claude 4.7+ (high-resolution tier), см. docs: build-with-claude/vision.
MAX_TOKENS = 4784
HIRES_MAX_EDGE = 2576
# Когда в разговоре больше 20 картинок, каждая должна быть ≤ 2000 px — поэтому это потолок по умолчанию.
DEFAULT_MAX_EDGE = 2000
# Обзорные картинки (лист целиком, силуэт, виды рядом, сравнение с моделью) — до ~1600 токенов (≈ 1120×1120 px):
# форму видно, а мелкое Claude смотрит через view. Так разговор дольше не забивается картинками.
OVERVIEW_TOKENS = 1600
# На картинках меньше ~200 px Claude ошибается.
MIN_EDGE = 200
# Мелкое увеличиваем: если длинная сторона меньше 768 px — до 1024 px, но не больше чем в 4 раза.
UPSCALE_BELOW = 768
UPSCALE_TARGET = 1024
MAX_UPSCALE = 4.0

JPEG_QUALITY = 92
# PNG выбираем, если он не больше чем в 2 раза тяжелее JPEG (линии, чертежи, плоские заливки).
PNG_VS_JPEG_RATIO = 2.0
# Потолок веса одной картинки (API принимает до 10 МБ в base64 ≈ 7.5 МБ сырых).
MAX_BYTES = 5_000_000
# Прозрачный фон заливаем нейтральным серым — на нём видны и светлые, и тёмные объекты.
NEUTRAL_GREY = (128, 128, 128)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


class ImageLoadError(Exception):
    """Картинку нельзя открыть; текст ошибки уходит модели как есть."""


@dataclass
class Prepared:
    data: bytes
    fmt: str  # "png" | "jpeg"
    size: tuple[int, int]
    flattened_alpha: bool


def count_image_tokens(width: int, height: int) -> int:
    """Визуальные токены Claude: один токен на блок 28×28 px."""
    return math.ceil(width / 28) * math.ceil(height / 28)


def resized_size(
    width: int,
    height: int,
    max_edge: int = HIRES_MAX_EDGE,
    max_tokens: int = MAX_TOKENS,
) -> tuple[int, int]:
    """Размер, до которого Claude сам ужал бы картинку (эталон из docs: vision-coordinates).

    Отдаём картинку ровно такого размера — тогда Claude её не пережимает и координаты не разъезжаются.
    """

    def fits(w: int, h: int) -> bool:
        return (
            math.ceil(w / 28) * 28 <= max_edge
            and math.ceil(h / 28) * 28 <= max_edge
            and count_image_tokens(w, h) <= max_tokens
        )

    if fits(width, height):
        return (width, height)
    if height > width:
        resized_h, resized_w = resized_size(height, width, max_edge, max_tokens)
        return (resized_w, resized_h)

    # Бинпоиск по длинной стороне: самый большой размер с теми же пропорциями, который влезает.
    aspect_ratio = width / height
    lo, hi = 1, width  # lo всегда влезает, hi — никогда
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / aspect_ratio), 1)):
            lo = mid
        else:
            hi = mid
    return (lo, max(round(lo / aspect_ratio), 1))


def fit_size(width: int, height: int, max_edge: int = DEFAULT_MAX_EDGE, max_tokens: int = MAX_TOKENS) -> tuple[int, int]:
    """Размер показа: мелкое увеличиваем, крупное ужимаем по правилам Claude."""
    long_edge = max(width, height)
    if long_edge < UPSCALE_BELOW:
        factor = min(MAX_UPSCALE, UPSCALE_TARGET / long_edge)
        width, height = max(1, round(width * factor)), max(1, round(height * factor))
    return resized_size(width, height, max_edge=max_edge, max_tokens=max_tokens)


def resolve_path(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        raise ImageLoadError(f"Нужен абсолютный путь, получено: {path}")
    if not p.exists():
        raise ImageLoadError(f"Файл не найден: {p}")
    if not p.is_file():
        raise ImageLoadError(f"Это не файл: {p}")
    return p


def open_image(p: Path) -> Image.Image:
    """Открывает картинку с учётом поворота из EXIF. У анимаций берётся первый кадр."""
    try:
        with Image.open(p) as img:
            img.load()
            fmt = img.format
            out = ImageOps.exif_transpose(img)
            out.format = fmt  # поворот по EXIF теряет формат файла
            return out
    except UnidentifiedImageError:
        raise ImageLoadError(f"Не картинка или формат не поддерживается: {p}") from None
    except Image.DecompressionBombError:
        raise ImageLoadError(f"Картинка слишком большая: {p}") from None
    except OSError as e:
        raise ImageLoadError(f"Не удалось прочитать картинку {p}: {e}") from None


def load_image(path: str) -> tuple[Image.Image, Path]:
    p = resolve_path(path)
    return open_image(p), p


def _has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)


def to_viewable(img: Image.Image) -> tuple[Image.Image, bool]:
    """Приводит к RGB или L. Прозрачность заливает серым. Возвращает (картинка, была_ли_прозрачность)."""
    if _has_alpha(img):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, NEUTRAL_GREY + (255,))
        return Image.alpha_composite(background, rgba).convert("RGB"), True
    if img.mode in ("RGB", "L"):
        return img, False
    return img.convert("RGB"), False


def _encode(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "jpeg":
        # subsampling=0 (4:4:4) — цветные края остаются чёткими.
        img.save(buf, "JPEG", quality=JPEG_QUALITY, subsampling=0)
    else:
        img.save(buf, "PNG")
    return buf.getvalue()


def _choose_format(img: Image.Image) -> str:
    """PNG для линий и плоских заливок, JPEG для фото и рендеров. Решаем по уменьшенной копии — быстро."""
    sample = img.copy()
    sample.thumbnail((512, 512))
    png, jpeg = _encode(sample, "png"), _encode(sample, "jpeg")
    return "png" if len(png) <= PNG_VS_JPEG_RATIO * len(jpeg) else "jpeg"


def render(
    img: Image.Image, max_edge: int = DEFAULT_MAX_EDGE, mode: str = "normal", max_tokens: int = MAX_TOKENS
) -> tuple[Image.Image, bool]:
    """Новая картинка размера показа в нужном режиме. Возвращает (картинка, была_ли_прозрачность).

    max_tokens — бюджет визуальных токенов: OVERVIEW_TOKENS для обзорных картинок."""
    viewable, flattened = to_viewable(img)
    target = fit_size(*viewable.size, max_edge=max_edge, max_tokens=max_tokens)
    if target != viewable.size:
        viewable = viewable.resize(target, Image.Resampling.LANCZOS)
    else:
        viewable = viewable.copy()
    return modes.apply(viewable, mode), flattened


def encode(img: Image.Image, force_png: bool = False) -> tuple[bytes, str]:
    fmt = "png" if force_png else _choose_format(img)
    data = _encode(img, fmt)
    if fmt == "png" and len(data) > MAX_BYTES:
        fmt, data = "jpeg", _encode(img, "jpeg")
    return data, fmt


def prepare(img: Image.Image, max_edge: int = DEFAULT_MAX_EDGE, mode: str = "normal", rulers: Box | None = None) -> Prepared:
    """Ужимает (или увеличивает мелкое) до точного размера, который Claude примет без пережатия, и кодирует.

    rulers — какая область оригинала на картинке: тогда по краям рисуются линейки в её пикселях."""
    shown, flattened = render(img, max_edge, mode)
    if rulers is not None:
        shown = measure.draw_rulers(shown, rulers)
    data, fmt = encode(shown, force_png=mode in modes.LINE_MODES)
    return Prepared(data=data, fmt=fmt, size=shown.size, flattened_alpha=flattened)


def file_size_text(n_bytes: int) -> str:
    if n_bytes >= 1024 * 1024:
        return f"{n_bytes / 1024 / 1024:.1f} МБ"
    return f"{max(1, round(n_bytes / 1024))} КБ"


def describe(
    path: Path,
    original_size: tuple[int, int],
    prepared: Prepared,
    region: list[int] | None = None,
    mode: str = "normal",
) -> str:
    """Текст к картинке: что показано и как перевести координаты обратно в пиксели оригинала."""
    ow, oh = original_size
    sw, sh = prepared.size
    x1, y1, x2, y2 = region if region else (0, 0, ow, oh)
    scale = (x2 - x1) / sw
    lines = [f"Файл: {path}"]
    if region:
        lines.append(f"Область оригинала [{x1}, {y1}, {x2}, {y2}] — {x2 - x1}×{y2 - y1} px (картинка {ow}×{oh} px)")
    else:
        lines.append(f"Оригинал: {ow}×{oh} px")
    lines.append(
        f"Показано: {sw}×{sh} px, {prepared.fmt.upper()}, {count_image_tokens(sw, sh)} визуальных токенов, "
        f"режим «{modes.MODE_NAMES[mode]}»"
    )
    if (sw, sh) == (x2 - x1, y2 - y1):
        zoom = "масштаб 1:1"
    elif scale < 1:
        zoom = f"увеличено в {1 / scale:.2f} раза"
    else:
        zoom = f"уменьшено в {scale:.2f} раза"
    if region:
        lines.append(f"Координаты в оригинале: x = {x1} + x_показ × {scale:.4f}, y = {y1} + y_показ × {scale:.4f} ({zoom}).")
    elif (sw, sh) == (ow, oh):
        lines.append("Масштаб 1:1 — координаты на картинке совпадают с пикселями оригинала.")
    else:
        lines.append(f"Координаты в оригинале: x = x_показ × {scale:.4f}, y = y_показ × {scale:.4f} ({zoom}).")
    if prepared.flattened_alpha:
        lines.append("Прозрачный фон залит серым (128, 128, 128).")
    return "\n".join(lines)
