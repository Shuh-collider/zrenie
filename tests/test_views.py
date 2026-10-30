import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from zrenie import matting, views
from zrenie.imaging import open_image
from zrenie.layout import Box, find_views

REFS = Path(__file__).parent / "refs"
BACKGROUND = (205, 207, 210)


def _object(size=(600, 1000), body=(220, 560), neck=(100, 240), colour=(90, 60, 40)):
    """Флакон на однотонном фоне: широкое тело внизу, узкое горлышко сверху — ключевая высота на стыке."""
    w, h = size
    img = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(img)
    cx = w // 2
    draw.rectangle([cx - body[0] // 2, h - 100 - body[1], cx + body[0] // 2, h - 100], fill=colour)
    draw.rectangle([cx - neck[0] // 2, h - 100 - body[1] - neck[1], cx + neck[0] // 2, h - 100 - body[1]], fill=colour)
    return img


def _source(img, name="вид.png"):
    return views.ViewSource(spec=name, path=Path(name), box=Box(0, 0, img.width, img.height), image=img, name=name, sheet_views=1)


def test_simple_matte_keeps_holes_and_measures():
    img = Image.new("RGB", (600, 600), BACKGROUND)
    ImageDraw.Draw(img).ellipse([100, 100, 500, 500], outline=(120, 90, 40), width=50)  # кольцо
    m = matting.simple_matte(img)
    shape = views.measure(m)
    assert shape.parts == 1 and shape.holes == 1
    assert abs(shape.bbox.x1 - 100) <= 3 and abs(shape.bbox.x2 - 501) <= 3
    assert abs(shape.aspect - 1.0) < 0.02
    assert not m.mask[300, 300]  # дырка кольца — фон


def test_key_height_at_neck_junction():
    m = matting.simple_matte(_object())
    shape = views.measure(m)
    # Тело 560 px из ~800 общей высоты объекта → стык на 70 % от низа.
    assert any(abs(k - 0.7) < 0.02 for k in shape.key_heights), shape.key_heights


@pytest.mark.skipif(not matting.ai_available(), reason="BiRefNet не установлен")
def test_ai_matte_excludes_shadow_on_real_reference():
    img = open_image(REFS / "single_headphones.jpg")
    shape = views.measure(matting.matte(img, "ai"))
    assert abs(shape.bbox.x1 - 149) <= 15 and abs(shape.bbox.x2 - 624) <= 15
    assert abs(shape.bbox.y1 - 81) <= 15
    assert shape.bbox.y2 < 905  # тень под наушниками (~y 900–930) не попала в объект


def test_warm_up_loads_birefnet_under_the_lock_and_never_fails(monkeypatch):
    loads = []
    monkeypatch.setattr(matting, "ai_available", lambda: True)
    monkeypatch.setattr(matting, "_load_model", lambda: loads.append(matting.GPU_LOCK.locked()))
    matting.warm_up()
    assert loads == [True]  # загрузка — под общей блокировкой нейросетей

    def broken():
        raise OSError("нет сети")

    monkeypatch.setattr(matting, "_load_model", broken)
    matting.warm_up()  # ошибка только в лог: сервер работает, модель загрузится при первом вызове


def test_parse_spec_forms(tmp_path):
    sheet_path = tmp_path / "лист 2×2.jpg"
    Image.open(REFS / "sheet_seams_4.jpg").save(sheet_path, quality=95)
    cache = {}

    def open_fn(path):
        p = Path(path)
        img = cache.setdefault(p, open_image(p))
        return img, p, (str(p),)

    def layout_fn(key):
        return find_views(cache[Path(key[0])])

    with pytest.raises(views.ViewError, match="лист из 4 кадров"):
        views.parse_spec(str(sheet_path), open_fn, layout_fn)
    third = views.parse_spec(f"{sheet_path}|3", open_fn, layout_fn)
    assert third.box.x1 == 0 and third.box.y1 >= 516 and "кадр 3" in third.name
    whole = views.parse_spec(f"{sheet_path}|0", open_fn, layout_fn)
    assert whole.box == Box(0, 0, 771, 1024)
    region = views.parse_spec(f"{sheet_path}|10, 20, 210, 320", open_fn, layout_fn)
    assert region.box == Box(10, 20, 210, 320) and region.image.size == (200, 300)
    with pytest.raises(views.ViewError, match="кадров 4"):
        views.parse_spec(f"{sheet_path}|7", open_fn, layout_fn)
    with pytest.raises(views.ViewError, match="Не понял область"):
        views.parse_spec(f"{sheet_path}|a,b", open_fn, layout_fn)


def _aligned(images_and_labels):
    result = []
    for img, label in images_and_labels:
        m = matting.simple_matte(img)
        result.append(views.AlignedView(_source(img), m, views.measure(m), views.normalize_label(label), views.label_text(label, "?"), 1.0))
    return result


def test_align_puts_objects_on_common_baseline_and_reports_mismatch():
    front = _object()
    back = _object(size=(900, 1500), body=(400, 840), neck=(150, 360))  # в 1.5 раза крупнее и тело шире на ~21 %
    aligned = _aligned([(front, "спереди"), (back, "back")])
    canvas, guides = views.compose_aligned(aligned)
    a, b = (v.placed for v in aligned)
    assert a.y2 == b.y2 and a.h == b.h == views.ALIGN_HEIGHT  # одна высота, общий низ
    assert any(abs(g - 0.7) < 0.02 for g in guides)
    problems = views.consistency_report(aligned)
    assert any("ширина спереди и сзади" in p for p in problems), problems


def test_export_writes_pngs_and_json(tmp_path):
    folder = tmp_path / "подложки"
    aligned = _aligned([(_object(), "front"), (_object(size=(500, 1000), body=(160, 560)), "side")])
    files, meta_path, meta = views.export(aligned, folder, real_height=0.2)
    assert [f.name for f in files] == ["01_front.png", "02_side.png"]
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    height = data["object_height_px"]
    assert abs(height - 800) <= 2 and abs(data["units_per_px"] - 0.2 / height) < 1e-9
    for record, path in zip(data["views"], files):
        img = Image.open(path)
        assert img.mode == "RGBA" and list(img.size) == record["image_px"]
        alpha = np.asarray(img)[..., 3]
        assert alpha[0, 0] == 0  # фон прозрачный
        x1, y1, x2, y2 = record["object_box_px"]
        assert abs((y2 - y1) - height) <= 1  # объект одной высоты
        assert alpha[round((y1 + y2) / 2), round((x1 + x2) / 2)] == 255
        assert record["image_center_from_object_bottom_center_units"][0] == pytest.approx(0, abs=0.002)
