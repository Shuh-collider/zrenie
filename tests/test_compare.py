from PIL import Image, ImageDraw

from zrenie import compare, matting

BACKGROUND = (205, 207, 210)


def _reference():
    """Флакон: тело 240×500 (x 180…419, y 350…849), горлышко 80×150 сверху."""
    img = Image.new("RGB", (600, 1000), BACKGROUND)
    draw = ImageDraw.Draw(img)
    draw.rectangle([180, 350, 419, 849], fill=(90, 60, 40))
    draw.rectangle([260, 200, 339, 349], fill=(110, 80, 50))
    return img


def _render(scale=1.3, body_half=120, neck_h=150):
    """«Рендер» с прозрачным фоном в другом масштабе сцены."""
    img = Image.new("RGBA", (900, 1500), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, bottom = 450, 1400
    draw.rectangle([cx - body_half * scale, bottom - 500 * scale, cx + body_half * scale - 1, bottom - 1], fill=(150, 150, 160, 255))
    draw.rectangle([cx - 40 * scale, bottom - (500 + neck_h) * scale, cx + 40 * scale - 1, bottom - 500 * scale], fill=(170, 170, 180, 255))
    return img


def _compare(render):
    ref = _reference()
    model_matte = compare.alpha_matte(render)
    assert model_matte is not None and "альфа" in model_matte.method
    return compare.compare(ref, matting.simple_matte(ref), render, model_matte)


def test_same_shape_in_other_scale_matches():
    result = _compare(_render())
    assert result.iou > 0.97
    assert abs(result.scale - 1 / 1.3) < 0.01
    assert not [f for f in result.findings if "шире" in f or "уже" in f]


def test_wider_body_and_taller_neck_are_reported():
    result = _compare(_render(body_half=150, neck_h=250))
    assert result.iou < 0.9
    text = "\n".join(result.findings)
    assert "модель шире" in text
    assert "Ключевые высоты сдвинуты" in text or "Не хватает" in text


def test_opaque_render_has_no_alpha_matte():
    assert compare.alpha_matte(_reference()) is None
    assert compare.alpha_matte(Image.new("RGBA", (10, 10), (1, 2, 3, 255))) is None


def test_panels_and_rows():
    ref = _reference()
    result = _compare(_render(body_half=150))
    for mode, count in (("all", 3), ("side", 2), ("diff", 1), ("overlay", 1)):
        assert len(compare.panels(ref, result, mode)) == count
    sheet = compare.compose_rows([("вид", compare.panels(ref, result, "all")), ("вид 2", compare.panels(ref, result, "all"))])
    assert sheet.width > 600 and sheet.height > 2 * 360  # две строки по три панели
