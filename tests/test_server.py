import base64
import io
import json
import re
import shutil
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters
from PIL import Image, ImageDraw

from zrenie import imaging
from zrenie.server import mcp

REFS = Path(__file__).parent / "refs"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def reference(tmp_path):
    folder = tmp_path / "референсы с пробелом"
    folder.mkdir()
    path = folder / "вид спереди.png"
    img = Image.new("RGB", (3000, 2000), "white")
    ImageDraw.Draw(img).rectangle([1000, 500, 2000, 1500], outline="black", width=8)
    img.save(path)
    return path


@pytest.fixture
def sheet_ref(tmp_path):
    """Настоящий лист 2×2 (кадры встык) в папке с кириллицей."""
    folder = tmp_path / "лист"
    folder.mkdir()
    path = folder / "наушники 2×2.jpg"
    shutil.copy(REFS / "sheet_seams_4.jpg", path)
    return path


def _image(block) -> Image.Image:
    assert block.type == "image"
    return Image.open(io.BytesIO(base64.b64decode(block.data)))


def _check_view_result(result):
    assert not result.is_error
    assert result.structured_content is None
    image, text = result.content
    assert image.mime_type in ("image/png", "image/jpeg")
    assert max(_image(image).size) <= 2000
    assert text.type == "text"
    assert "Оригинал: 3000×2000 px" in text.text


@pytest.mark.anyio
async def test_tools_are_listed():
    async with Client(mcp) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert {"list_media", "inspect", "split_views", "view", "sheet"} <= names


@pytest.mark.anyio
async def test_view_in_process(reference):
    async with Client(mcp) as client:
        _check_view_result(await client.call_tool("view", {"path": str(reference)}))


@pytest.mark.anyio
async def test_view_region_is_upscaled_and_maps_back(sheet_ref):
    async with Client(mcp) as client:
        result = await client.call_tool("view", {"path": str(sheet_ref), "region": [150, 860, 384, 1024]})
    assert not result.is_error
    image, text = result.content
    assert max(_image(image).size) == 936  # 234 px × 4 — мелкая область увеличена
    assert "x = 150 + x_показ × 0.2500" in text.text
    assert "Линейки" in text.text


@pytest.mark.anyio
async def test_overview_is_smaller_than_view(reference):
    """Обзор целиком — в бюджете обзорных картинок, крупный вид — в полном размере."""
    async with Client(mcp) as client:
        overview = await client.call_tool("inspect", {"path": str(reference)})
        detail = await client.call_tool("view", {"path": str(reference)})
    assert imaging.count_image_tokens(*_image(overview.content[0]).size) <= imaging.OVERVIEW_TOKENS
    assert imaging.count_image_tokens(*_image(detail.content[0]).size) > 2 * imaging.OVERVIEW_TOKENS


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["contrast", "edges", "tones"])
async def test_view_modes(sheet_ref, mode):
    async with Client(mcp) as client:
        result = await client.call_tool("view", {"path": str(sheet_ref), "region": [0, 516, 384, 1024], "mode": mode})
    assert not result.is_error
    image = result.content[0]
    assert image.mime_type == ("image/jpeg" if mode == "contrast" else "image/png")


@pytest.mark.anyio
async def test_inspect_finds_views_on_sheet(sheet_ref):
    async with Client(mcp) as client:
        result = await client.call_tool("inspect", {"path": str(sheet_ref)})
    assert not result.is_error
    image, text = result.content
    assert _image(image).size == (771, 1024)
    assert "найдено кадров — 4" in text.text
    assert text.text.count("Кадр ") == 4


@pytest.mark.anyio
async def test_split_views_returns_label_then_image_for_each_view(sheet_ref):
    async with Client(mcp) as client:
        result = await client.call_tool("split_views", {"path": str(sheet_ref)})
    assert not result.is_error
    blocks = result.content
    assert [b.type for b in blocks] == ["text", "image"] * 4 + ["text"]
    for label, image in zip(blocks[0:8:2], blocks[1:8:2]):
        assert label.text.startswith("Кадр ")
        assert max(_image(image).size) == 1008  # кадры ~500 px увеличены до лимита 1024 (кратно 28)


@pytest.mark.anyio
async def test_sheet_and_list_media(sheet_ref, reference, tmp_path):
    async with Client(mcp) as client:
        result = await client.call_tool("sheet", {"paths": [str(sheet_ref), str(reference)], "labels": ["лист", "вид"]})
        assert not result.is_error
        image, text = result.content
        assert max(_image(image).size) <= 2000
        assert "лист — на листе" in text.text and "вид — на листе" in text.text

        listing = await client.call_tool("list_media", {"folder": str(tmp_path), "recursive": True})
        assert not listing.is_error
        assert "наушники 2×2.jpg — 771×1024 px" in listing.content[0].text


@pytest.mark.anyio
async def test_errors_are_readable(tmp_path, reference):
    async with Client(mcp) as client:
        missing = await client.call_tool("view", {"path": str(tmp_path / "нет.png")})
        assert missing.is_error
        assert "не найден" in missing.content[0].text

        bad_edge = await client.call_tool("view", {"path": str(reference), "max_edge": 5000})
        assert bad_edge.is_error

        bad_mode = await client.call_tool("view", {"path": str(reference), "mode": "xray"})
        assert bad_mode.is_error

        outside = await client.call_tool("view", {"path": str(reference), "region": [5000, 5000, 6000, 6000]})
        assert outside.is_error
        assert "вне картинки" in outside.content[0].text

        bad_labels = await client.call_tool("sheet", {"paths": [str(reference)], "labels": ["a", "b"]})
        assert bad_labels.is_error


@pytest.mark.anyio
async def test_view_over_real_stdio(reference):
    """Тот же вызов через настоящий процесс `python -m zrenie` — как его запускает Claude Code."""
    params = StdioServerParameters(command=sys.executable, args=["-m", "zrenie"])
    async with Client(params) as client:
        _check_view_result(await client.call_tool("view", {"path": str(reference)}))


@pytest.fixture
def bottle(tmp_path):
    """Простой «флакон» на однотонном фоне — вид спереди и сбоку, в папке с кириллицей."""
    folder = tmp_path / "виды"
    folder.mkdir()
    paths = []
    for name, body in (("спереди.png", 220), ("сбоку.png", 150)):
        img = Image.new("RGB", (600, 800), (205, 207, 210))
        draw = ImageDraw.Draw(img)
        draw.rectangle([300 - body // 2, 140, 300 + body // 2, 700], fill=(90, 60, 40))
        draw.rectangle([250, 40, 350, 140], fill=(90, 60, 40))
        img.save(folder / name)
        paths.append(str(folder / name))
    return paths


@pytest.mark.anyio
async def test_silhouette_align_export(bottle, sheet_ref, tmp_path):
    async with Client(mcp) as client:
        sil = await client.call_tool("silhouette", {"view": bottle[0], "method": "simple"})
        assert not sil.is_error
        aspect = float(re.search(r"ширина / высота = ([\d.]+)", sil.content[1].text).group(1))
        assert abs(aspect - 221 / 661) < 0.01

        aligned = await client.call_tool("align_views", {"views": bottle, "labels": ["front", "side"], "method": "simple"})
        assert not aligned.is_error
        assert aligned.content[0].type == "image" and "«спереди»" in aligned.content[1].text

        sheet_needs_frame = await client.call_tool("align_views", {"views": [str(sheet_ref)], "method": "simple"})
        assert sheet_needs_frame.is_error and "|1" in sheet_needs_frame.content[0].text

        out = tmp_path / "экспорт"
        exported = await client.call_tool(
            "export_views", {"views": bottle, "folder": str(out), "labels": ["front", "side"], "real_height": 0.25, "method": "simple"}
        )
        assert not exported.is_error
        assert (out / "01_front.png").exists() and (out / "02_side.png").exists() and (out / "views.json").exists()

        bad = await client.call_tool("export_views", {"views": bottle, "folder": "relative", "method": "simple"})
        assert bad.is_error


@pytest.mark.anyio
async def test_grid_mark_contour_palette(bottle, tmp_path):
    front = bottle[0]  # тело 221×561 px (x 190…410, y 140…700), горлышко 101×101 сверху; ось на x = 300.5
    async with Client(mcp) as client:
        px = await client.call_tool("grid", {"view": front})
        assert not px.is_error and "через 100 px" in px.content[1].text

        pct = await client.call_tool("grid", {"view": front, "units": "percent", "method": "simple"})
        assert not pct.is_error and "1 % высоты = 6.61 px" in pct.content[1].text

        marked = await client.call_tool(
            "mark",
            {"view": front, "points": [[415, 400]], "lines": [[195, 650, 405, 650]], "snap": "silhouette", "method": "simple"},
        )
        assert not marked.is_error
        text = marked.content[1].text
        assert "P1: (410.0, 400.0)" in text  # точка притянута к правому краю тела
        assert "длина 221.0 px" in text  # концы линии притянуты вдоль неё к краям тела шириной 221 px

        out = tmp_path / "контур" / "флакон.json"
        cont = await client.call_tool("contour", {"view": front, "half": True, "save_to": str(out), "method": "simple"})
        assert not cont.is_error and "симметричен" in cont.content[1].text
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["contours"][0]["kind"] == "outer" and abs(data["axis_x_px"] - 300.5) <= 1
        profile = data["half_profile"]
        assert profile[0][0] == pytest.approx(0, abs=1e-4) and profile[-1][0] == pytest.approx(0, abs=1e-4)

        colours = await client.call_tool("palette", {"view": front, "colors": 2, "method": "simple"})
        assert not colours.is_error and "#5A3C28" in colours.content[1].text  # (90, 60, 40)


@pytest.mark.anyio
async def test_compare_and_compare_views(bottle, tmp_path):
    render = tmp_path / "рендер спереди.png"
    img = Image.new("RGBA", (500, 900), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 160, 400, 800], fill=(120, 120, 130, 255))  # тело шире, чем у «флакона», относительно высоты
    draw.rectangle([200, 40, 300, 160], fill=(120, 120, 130, 255))
    img.save(render)
    async with Client(mcp) as client:
        one = await client.call_tool("compare", {"reference": bottle[0], "render": str(render), "method": "simple"})
        assert not one.is_error
        text = one.content[1].text
        assert "Совпадение силуэтов:" in text and "альфа-канал рендера" in text and "модель шире" in text

        for mode in ("side", "diff", "overlay"):
            r = await client.call_tool("compare", {"reference": bottle[0], "render": str(render), "mode": mode, "method": "simple"})
            assert not r.is_error

        many = await client.call_tool(
            "compare_views",
            {"references": bottle, "renders": [str(render), str(render)], "labels": ["front", "side"], "method": "simple"},
        )
        assert not many.is_error and "Среднее совпадение:" in many.content[1].text and "«спереди»" in many.content[1].text

        bad = await client.call_tool("compare_views", {"references": bottle, "renders": [str(render)], "method": "simple"})
        assert bad.is_error
