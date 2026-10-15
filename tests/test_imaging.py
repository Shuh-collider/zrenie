import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from zrenie import imaging


@pytest.mark.parametrize(
    ("size", "limits", "expected"),
    [
        # Примеры из документации Claude (vision, vision-coordinates).
        ((1075, 1520), (1568, 1568), (924, 1307)),
        ((1920, 1080), (1568, 1568), (1456, 819)),
        ((3840, 2160), (1568, 1568), (1456, 819)),
        ((3840, 2160), (2576, 4784), (2576, 1449)),
        ((1920, 1080), (2576, 4784), (1920, 1080)),
        ((2000, 1500), (2576, 4784), (2000, 1500)),
        ((1000, 1000), (1568, 1568), (1000, 1000)),
    ],
)
def test_resized_size_matches_docs(size, limits, expected):
    assert imaging.resized_size(*size, max_edge=limits[0], max_tokens=limits[1]) == expected


@pytest.mark.parametrize("size", [(4096, 4096), (6000, 1000), (1000, 6000), (2048, 2048)])
def test_default_limits_keep_under_2000px_and_token_budget(size):
    w, h = imaging.resized_size(*size, max_edge=imaging.DEFAULT_MAX_EDGE)
    assert max(w, h) <= 2000
    assert imaging.count_image_tokens(w, h) <= imaging.MAX_TOKENS


def _line_art(size=(1200, 800)):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    for x in range(0, size[0], 40):
        draw.line([(x, 0), (size[0] - x, size[1])], fill="black", width=2)
    return img


def _photo_like(size=(1200, 800)):
    rng = np.random.default_rng(0)
    gradient = np.linspace(0, 255, size[0], dtype=np.float32)[None, :, None]
    noise = rng.normal(0, 25, (size[1], size[0], 3))
    return Image.fromarray(np.clip(gradient + noise, 0, 255).astype(np.uint8))


def test_line_art_goes_png_and_photo_goes_jpeg():
    assert imaging.prepare(_line_art()).fmt == "png"
    assert imaging.prepare(_photo_like()).fmt == "jpeg"


def test_prepare_resizes_large_image_to_exact_target():
    prepared = imaging.prepare(_photo_like((4000, 3000)))
    assert prepared.size == imaging.resized_size(4000, 3000, max_edge=2000)
    assert Image.open(io.BytesIO(prepared.data)).size == prepared.size


def test_prepare_keeps_small_image_one_to_one():
    assert imaging.prepare(_line_art((800, 600))).size == (800, 600)


def test_transparent_background_is_flattened_to_grey():
    img = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    prepared = imaging.prepare(img)
    assert prepared.flattened_alpha
    decoded = Image.open(io.BytesIO(prepared.data)).convert("RGB")
    assert decoded.getpixel((10, 10)) == imaging.NEUTRAL_GREY


def test_load_image_in_cyrillic_path_with_spaces(tmp_path):
    folder = tmp_path / "папка с пробелом"
    folder.mkdir()
    path = folder / "референс спереди.png"
    _line_art().save(path)
    img, p = imaging.load_image(str(path))
    assert img.size == (1200, 800)
    assert p == path


def test_load_image_errors(tmp_path):
    with pytest.raises(imaging.ImageLoadError, match="абсолютный"):
        imaging.load_image("relative.png")
    with pytest.raises(imaging.ImageLoadError, match="не найден"):
        imaging.load_image(str(tmp_path / "нет.png"))
    not_image = tmp_path / "текст.png"
    not_image.write_text("это не картинка", encoding="utf-8")
    with pytest.raises(imaging.ImageLoadError, match="Не картинка"):
        imaging.load_image(str(not_image))


def test_describe_reports_scale():
    prepared = imaging.prepare(_photo_like((4000, 3000)))
    text = imaging.describe(Path("C:/x.png"), (4000, 3000), prepared)
    assert "Оригинал: 4000×3000 px" in text
    assert f"{4000 / prepared.size[0]:.4f}" in text
