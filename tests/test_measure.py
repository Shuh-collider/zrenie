import numpy as np
import pytest
from PIL import Image, ImageDraw

from zrenie import matting, measure, views
from zrenie.layout import Box

BACKGROUND = (205, 207, 210)


def _bottle(offset_x=0):
    """Флакон: тело 240×500, горлышко 80×150, ось на x = 300 + offset_x; низ на y = 850."""
    img = Image.new("RGB", (600 + 2 * abs(offset_x), 1000), BACKGROUND)
    draw = ImageDraw.Draw(img)
    cx = 300 + offset_x
    draw.rectangle([cx - 120, 350, cx + 119, 849], fill=(90, 60, 40))
    draw.rectangle([cx - 40, 200, cx + 39, 349], fill=(90, 60, 40))
    return img


@pytest.mark.parametrize(("span", "step"), [(1152, 200), (920, 100), (480, 50), (130, 20), (9, 1)])
def test_nice_step(span, step):
    assert measure.nice_step(span) == step


def test_rulers_are_in_original_pixels():
    shown = Image.new("RGB", (1000, 500), "white")
    out = measure.draw_rulers(shown, Box(2000, 1000, 4000, 2000))  # 2 px оригинала на пиксель показа
    assert out.size == shown.size
    a = np.asarray(out)
    magenta = np.all(a == measure.GRID_COLOUR, axis=2)
    assert magenta[2, 99:102].any()  # деление x = 2200: (2200 − 2000) / 2 = 100 px показа
    assert magenta[99:102, 2].any()  # деление y = 1200: (1200 − 1000) / 2 = 100 px показа
    assert not magenta[2, 145:155].any()  # между делениями картинка как была
    assert (a[8:24, 85:115] < 100).any()  # подпись «2200» под делением
    assert (a[300:, 100:] == 255).all()  # середина картинки не тронута


def test_symmetry_axis_and_half_profile():
    img = _bottle(offset_x=37)
    mask = matting.simple_matte(img).mask
    shape = views.measure(mask_matte(mask))
    sym = measure.find_symmetry(mask, shape.bbox)
    assert sym.score > 0.97
    assert abs(sym.axis_x - 337) <= 1.5  # ось между пикселями 336 и 337

    outer = measure.find_contours(mask, shape.bbox, measure.CONTOUR_DETAIL)[0]
    profile = measure.half_profile(outer.points, sym.axis_x)
    scene = measure.to_scene(profile, sym.axis_x, shape.bbox.y2, 1 / shape.bbox.h)
    assert abs(scene[0, 0]) < 1e-6 and abs(scene[-1, 0]) < 1e-6  # начинается и кончается на оси
    assert scene[0, 1] > scene[-1, 1]  # сверху вниз
    assert (scene[:, 0] >= -1e-6).all()
    assert abs(scene[:, 0].max() - 120 / 650) < 0.01  # полуширина тела в высотах объекта


def mask_matte(mask):
    return matting.Matte(mask.astype(np.float32), "тест")


def test_ring_contour_has_hole():
    img = Image.new("RGB", (600, 600), BACKGROUND)
    ImageDraw.Draw(img).ellipse([100, 100, 500, 500], outline=(120, 90, 40), width=50)
    mask = matting.simple_matte(img).mask
    contours = measure.find_contours(mask, views.measure(mask_matte(mask)).bbox, measure.CONTOUR_DETAIL)
    assert [c.kind for c in contours] == ["outer", "hole"]


def test_snap_to_silhouette_edge():
    img = _bottle()
    mask = matting.simple_matte(img).mask
    results = measure.snap_points([(426.0, 600.0), (300.0, 100.0)], img, (0, 0), mask, "silhouette")
    near, far = results
    assert near.snapped is not None and abs(near.snapped[0] - 419) <= 1 and near.distance <= 8
    assert far.snapped is None  # края рядом нет — точка остаётся как есть


def test_palette_two_colours():
    img = Image.new("RGB", (400, 200), (200, 30, 30))
    ImageDraw.Draw(img).rectangle([300, 0, 399, 199], fill=(20, 40, 220))
    swatches = measure.palette(img, None, 2)
    assert max(abs(a - b) for a, b in zip(swatches[0].rgb, (200, 30, 30))) <= 2 and abs(swatches[0].share - 0.75) < 0.02
    assert max(abs(a - b) for a, b in zip(swatches[1].rgb, (20, 40, 220))) <= 2
    assert swatches[0].linear01[0] == pytest.approx(0.578, abs=0.01)


@pytest.mark.parametrize("mode", ["edges", "silhouette"])
def test_line_ends_snap_along_the_line_only(mode):
    img = _bottle()  # тело: x 180…419, y 350…849
    mask = matting.simple_matte(img).mask
    # Горизонтальная линия поперёк тела с концами мимо краёв на 6 px; рядом сбоку — горлышко, на него прыгать нельзя.
    (x1, y1, x2, y2), shifts = measure.snap_line((186.0, 360.0, 413.0, 360.0), img, (0, 0), mask, mode)
    assert y1 == y2 == 360.0  # концы двигались только вдоль линии
    assert abs(x1 - 180) <= 1.0 and abs(x2 - 420) <= 1.0
    assert all(s is not None for s in shifts)
    # Вертикальная линия по высоте тела.
    (_, top, _, bottom), _ = measure.snap_line((250.0, 358.0, 250.0, 840.0), img, (0, 0), mask, mode)
    assert abs(top - 350) <= 1.0 and abs(bottom - 850) <= 1.0
