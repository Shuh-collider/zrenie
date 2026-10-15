from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from zrenie.layout import Box, find_views

REFS = Path(__file__).parent / "refs"


def _photo(size, seed, base, wide=False):
    """Кадр «как фото»: плавный фон своего цвета, объект посередине, шум поверх всего.

    wide=True — объект шире кадра и обрезан его краями, поэтому между кадрами нет пустых полей.
    """
    w, h = size
    gradient = np.linspace(-20, 20, h, dtype=np.float32)[:, None, None]
    background = np.array(base, np.float32)[None, None] + gradient + np.zeros((h, w, 1), np.float32)
    img = Image.fromarray(np.clip(background, 0, 255).astype(np.uint8))
    box = [-0.2 * w, 0.15 * h, 1.2 * w, 0.85 * h] if wide else [w * 0.25, h * 0.2, w * 0.75, h * 0.85]
    fill = (90 + 30 * seed, 70 + 15 * seed, 60)
    ImageDraw.Draw(img).ellipse(box, fill=fill, outline=(20, 20, 20), width=4)
    noise = np.random.default_rng(seed).normal(0, 3, (h, w, 3))
    return Image.fromarray(np.clip(np.asarray(img, np.float32) + noise, 0, 255).astype(np.uint8))


def _centers_match(boxes: list[Box], expected: list[tuple[float, float]], size, tol=0.08):
    w, h = size
    centers = [((b.x1 + b.x2) / 2 / w, (b.y1 + b.y2) / 2 / h) for b in boxes]
    assert len(centers) == len(expected)
    for (cx, cy), (ex, ey) in zip(centers, expected):
        assert abs(cx - ex) < tol and abs(cy - ey) < tol, (centers, expected)


def test_grid_with_white_gutters():
    sheet = Image.new("RGB", (812, 812), "white")
    for i, base in enumerate([(180, 190, 200), (150, 160, 170), (200, 200, 190), (120, 130, 150)]):
        row, col = divmod(i, 2)
        sheet.paste(_photo((400, 400), i, base), (col * 412, row * 412))
    layout = find_views(sheet)
    assert layout.method == "промежутки"
    _centers_match(layout.views, [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)], sheet.size)


def test_grid_glued_without_gutters_is_split_by_seams():
    sheet = Image.new("RGB", (800, 800))
    for i, base in enumerate([(180, 190, 200), (150, 160, 170), (200, 200, 190), (120, 130, 150)]):
        row, col = divmod(i, 2)
        sheet.paste(_photo((400, 400), i, base, wide=True), (col * 400, row * 400))
    layout = find_views(sheet)
    assert "швы" in layout.method
    _centers_match(layout.views, [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)], sheet.size)
    # Каждый кадр — внутри своей ячейки 400×400 и во всю её ширину (объект шире кадра).
    for box in layout.views:
        cell_x, cell_y = (box.x1 + 10) // 400 * 400, (box.y1 + 10) // 400 * 400
        assert cell_x - 5 <= box.x1 and box.x2 <= cell_x + 405 and cell_y - 5 <= box.y1 and box.y2 <= cell_y + 405
        assert box.w >= 390


def test_turnaround_on_shared_background_drops_text_labels():
    sheet = Image.new("RGB", (1500, 700), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    for i in range(3):
        x = 100 + i * 480
        draw.rounded_rectangle([x, 80, x + 300, 560], radius=60, fill=(70, 80, 90), outline=(10, 10, 10), width=4)
        draw.rectangle([x + 100, 610, x + 200, 650], fill=(20, 20, 20))  # «подпись» под видом
    layout = find_views(sheet)
    assert len(layout.views) == 3
    assert layout.dropped >= 1
    _centers_match(layout.views, [(250 / 1500, 0.46), (730 / 1500, 0.46), (1210 / 1500, 0.46)], sheet.size)


def test_single_object_with_straight_center_line_is_not_split():
    img = Image.new("RGB", (800, 1000), (230, 232, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle([200, 150, 600, 850], fill=(120, 100, 80), outline=(20, 20, 20), width=4)
    draw.line([(400, 150), (400, 850)], fill=(20, 20, 20), width=3)  # шов дверцы ровно посередине
    layout = find_views(img)
    assert layout.method == "один кадр"
    assert len(layout.views) == 1
    c = layout.content
    assert abs(c.x1 - 200) < 15 and abs(c.x2 - 601) < 15 and abs(c.y1 - 150) < 15 and abs(c.y2 - 851) < 15


@pytest.mark.parametrize(
    ("name", "count", "method"),
    [
        ("sheet_gutters_5.jpg", 5, "промежутки"),
        ("sheet_seams_4.jpg", 4, "промежутки и швы"),
        ("single_headphones.jpg", 1, "один кадр"),
        ("renders_grid_3x2.jpg", 6, "швы (кадры встык)"),
        # Во втором кадре чёткий горизонт на 2/3 высоты — его нельзя принять за шов.
        ("renders_strip_1x4.jpg", 4, "швы (кадры встык)"),
    ],
)
def test_real_references(name, count, method):
    layout = find_views(Image.open(REFS / name))
    assert len(layout.views) == count
    assert layout.method == method


def test_seam_cut_has_no_strip_of_the_neighbour_frame():
    """На листе встык граница кадров 3|4 — на x=386; смешанные пиксели шва не попадают ни в один кадр."""
    views = find_views(Image.open(REFS / "sheet_seams_4.jpg")).views
    assert views[2].x2 <= 385 and views[3].x1 >= 387
    assert views[2].y1 >= 517 and views[0].y2 <= 515
