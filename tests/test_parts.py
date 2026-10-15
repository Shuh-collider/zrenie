import json
import re

import numpy as np
import pytest
from mcp import Client
from PIL import Image, ImageDraw

from zrenie import matting, parts as pt, server
from zrenie import views as vs
from zrenie.server import mcp

BACKGROUND = (205, 207, 210)
BODY, CAP = (90, 60, 40), (40, 40, 45)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def bottle(tmp_path):
    """«Флакон»: тело 221×561 px (x 190…410, y 140…700) и крышка 101×101 сверху (x 250…350, y 40…140)."""
    folder = tmp_path / "детали"
    folder.mkdir()
    path = folder / "флакон спереди.png"
    img = Image.new("RGB", (600, 800), BACKGROUND)
    draw = ImageDraw.Draw(img)
    draw.rectangle([190, 140, 410, 700], fill=BODY)
    draw.rectangle([250, 40, 350, 139], fill=CAP)
    img.save(path)
    return path


def _remember(path, masks: dict[str, np.ndarray]) -> None:
    """Детали в памяти сервера без нейросети — как будто их выделил parts."""
    source = server._source(str(path))
    server._parts[server._view_key(source)] = {name: pt.Part(name, m, 1.0) for name, m in masks.items()}


def _rect(shape, x1, y1, x2, y2) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[y1:y2, x1:x2] = True
    return m


def test_spec_with_part(bottle):
    open_ = server._open
    layout = lambda key: server._cached_layout(*key)  # noqa: E731
    s = vs.parse_spec(f"{bottle}|крышка", open_, layout)
    assert s.part == "крышка" and s.box.as_list() == [0, 0, 600, 800] and "деталь «крышка»" in s.name
    s = vs.parse_spec(f"{bottle}|10,20,300,400|крышка", open_, layout)
    assert s.part == "крышка" and s.box.as_list() == [10, 20, 300, 400]
    s = vs.parse_spec(f"{bottle}|0", open_, layout)
    assert s.part is None
    with pytest.raises(vs.ViewError):
        vs.parse_spec(f"{bottle}|крышка|тело", open_, layout)


def test_measure_part_angle_and_colour():
    img = Image.new("RGB", (400, 400), BACKGROUND)
    draw = ImageDraw.Draw(img)
    # Палка, наклонённая на 30° к горизонтали против часовой (правый конец выше).
    draw.line([(100, 300), (100 + 200 * np.cos(np.radians(30)), 300 - 200 * np.sin(np.radians(30)))], fill=BODY, width=20)
    mask = np.asarray(img) [..., 0] < 150
    shape = pt.measure_part(mask, img)
    assert shape.angle == pytest.approx(30, abs=1.5)
    assert shape.colour == "#5A3C28"
    square = pt.measure_part(_rect((100, 100), 20, 20, 80, 80))
    assert square.angle is None and square.holes == 0 and square.pieces == 1


def test_model_parts_from_id_map(tmp_path):
    img = Image.new("RGBA", (50, 40), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([5, 5, 19, 34], fill=(230, 25, 75, 255))
    draw.rectangle([25, 5, 44, 14], fill=(62, 178, 77, 255))  # чуть другой оттенок — в пределах допуска
    path = tmp_path / "карта.png"
    img.save(path)
    path.with_suffix(".json").write_text(json.dumps({"objects": {"тело": "#E6194B", "крышка": "#3CB44B", "нет": "#0082C8"}}), encoding="utf-8")
    masks = pt.model_parts(img, pt.read_id_map(path))
    assert masks["тело"].sum() == 15 * 30 and masks["крышка"].sum() == 20 * 10 and not masks["нет"].any()


@pytest.mark.anyio
async def test_part_view_in_other_tools(bottle, tmp_path):
    _remember(bottle, {"крышка": _rect((800, 600), 250, 40, 351, 140)})
    async with Client(mcp) as client:
        sil = await client.call_tool("silhouette", {"view": f"{bottle}|крышка"})
        assert not sil.is_error
        assert "[250, 40, 351, 140] — 101×100 px" in sil.content[1].text and "деталь «крышка»" in sil.content[1].text

        colours = await client.call_tool("palette", {"view": f"{bottle}|крышка", "colors": 1})
        assert not colours.is_error and "#28282D" in colours.content[1].text

        missing = await client.call_tool("contour", {"view": f"{bottle}|ручка"})
        assert missing.is_error and "крышка" in missing.content[0].text

        whole = await client.call_tool("compare", {"reference": f"{bottle}|крышка", "render": str(bottle)})
        assert whole.is_error and "compare_parts" in whole.content[0].text


@pytest.mark.anyio
async def test_compare_parts(bottle, tmp_path):
    _remember(bottle, {"тело": _rect((800, 600), 190, 140, 411, 701), "крышка": _rect((800, 600), 250, 40, 351, 140)})
    # Модель в 1,25 раза крупнее, крышка сдвинута вправо на 25 px (20 px в масштабе референса = 9 % ширины).
    render = tmp_path / "front_parts.png"
    img = Image.new("RGBA", (525, 1000), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([125, 175, 400 - 1, 875 - 1], fill=(230, 25, 75, 255))
    draw.rectangle([225, 50, 350 - 1, 175 - 1], fill=(60, 180, 75, 255))
    img.save(render)
    render.with_suffix(".json").write_text(json.dumps({"objects": {"тело": "#E6194B", "крышка": "#3CB44B"}}), encoding="utf-8")

    async with Client(mcp) as client:
        result = await client.call_tool("compare_parts", {"reference": str(bottle), "render": str(render), "method": "simple"})
        assert not result.is_error
        text = result.content[1].text
        body = re.search(r"«тело».*?: ([\d.]+) %", text)
        cap = re.search(r"«крышка».*?: ([\d.]+) % — (.*)", text)
        assert float(body.group(1)) > 97
        assert float(cap.group(1)) < 80 and "правее на 9." in cap.group(2)

        named = await client.call_tool(
            "compare_parts", {"reference": str(bottle), "render": str(render), "match": {"крышка": ["нет"]}, "method": "simple"}
        )
        assert named.is_error and "тело" in named.content[0].text


def _sam_ready() -> bool:
    if not matting.ai_available():
        return False
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(pt.SAM_REPO, "model.safetensors", revision=pt.SAM_REVISION), str)


@pytest.mark.anyio
@pytest.mark.skipif(not _sam_ready(), reason="SAM 2.1 не установлен")
async def test_parts_with_sam(bottle):
    async with Client(mcp) as client:
        result = await client.call_tool(
            "parts",
            {"view": str(bottle), "parts": [{"name": "крышка", "points": [[300, 90]]}, {"name": "тело", "box": [180, 130, 420, 710]}], "method": "simple"},
        )
        assert not result.is_error
        text = result.content[1].text
        cap = re.search(r"«крышка».*?рамка в оригинале \[(\d+), (\d+), (\d+), (\d+)\]", text)
        assert np.allclose([int(v) for v in cap.groups()], [250, 40, 351, 140], atol=4)
        assert "Варианты по одной точке" in text

        again = await client.call_tool("contour", {"view": f"{bottle}|тело", "method": "simple"})
        assert not again.is_error and "симметричен" in again.content[1].text
