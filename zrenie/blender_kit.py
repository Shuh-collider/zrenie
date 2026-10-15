"""Заготовка для Blender: «карта деталей» для compare_parts — каждый объект залит своим плоским цветом.

Запускается внутри Blender, сервер этот файл не импортирует. Через execute_blender_code:

    exec(open(r"<папка проекта>\\zrenie\\blender_kit.py", encoding="utf-8").read())
    result = render_parts("cam_front", r"C:\\путь\\front_parts.png")

Рядом с PNG пишется JSON {"objects": {"имя объекта": "#RRGGBB"}} — по нему сервер вырезает детали модели.
Настройки сцены и цвета объектов после рендера возвращаются как были.
"""

import json
from pathlib import Path

import bpy
import numpy as np

RENDERABLE = {"MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES", "POINTCLOUD", "VOLUME"}
CHECK_TOLERANCE = 12  # как ID_TOLERANCE в zrenie/parts.py
MIN_DISTANCE = 40  # цвета объектов отличаются хотя бы на столько по одному из каналов


def _palette():
    """Сначала хорошо различимые на глаз цвета, потом — сетка остальных; любые два отличаются на MIN_DISTANCE."""
    base = [
        "#E6194B", "#3CB44B", "#0082C8", "#F58230", "#911EB4", "#46F0F0", "#F032E6", "#D2F53C", "#008080", "#AA6E28",
        "#800000", "#000080", "#FFE119", "#808000", "#FABED4", "#AAFFC3", "#DCBEFF", "#FFD7B4", "#808080", "#FFFAC8",
    ]
    chosen = [tuple(int(h[i : i + 2], 16) for i in (1, 3, 5)) for h in base]
    levels = (0, 64, 128, 192, 255)
    for r in levels:
        for g in levels:
            for b in levels:
                if all(max(abs(r - c[0]), abs(g - c[1]), abs(b - c[2])) >= MIN_DISTANCE for c in chosen):
                    chosen.append((r, g, b))
    return [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in chosen]


PART_HEX = _palette()


def _linear(hex_colour):
    """sRGB HEX → линейный цвет: Object.color в Blender линейный, а вид Standard переводит его обратно в sRGB."""
    out = []
    for i in (1, 3, 5):
        c = int(hex_colour[i : i + 2], 16) / 255
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return (*out, 1.0)


def _targets(scene, objects):
    if objects is None:
        return sorted((o for o in scene.objects if o.type in RENDERABLE and not o.hide_render), key=lambda o: o.name)
    if isinstance(objects, str):
        collection = bpy.data.collections.get(objects)
        if collection is None:
            raise ValueError(f"Коллекции «{objects}» нет")
        chosen = collection.all_objects
    else:
        missing = [n for n in objects if n not in bpy.data.objects]
        if missing:
            raise ValueError(f"Объектов нет: {', '.join(missing)}")
        chosen = [bpy.data.objects[n] for n in objects]
    return sorted((o for o in chosen if o.type in RENDERABLE and not o.hide_render), key=lambda o: o.name)


def render_parts(camera, path, objects=None, resolution=None):
    """Рендер карты деталей с камеры camera в PNG path (+ JSON рядом).

    objects: None — все видимые в рендере объекты; имя коллекции; или список имён объектов (остальные скрываются).
    resolution: (ширина, высота) или None — как в сцене. Лучше то же, что у рендеров для compare.
    Возвращает словарь: файлы, цвет каждого объекта и сколько его пикселей видно.
    """
    scene = bpy.context.scene
    cam = bpy.data.objects.get(camera)
    if cam is None or cam.type != "CAMERA":
        raise ValueError(f"Камеры «{camera}» нет")
    targets = _targets(scene, objects)
    if len(targets) > len(PART_HEX):
        raise ValueError(f"Объектов {len(targets)}, а цветов {len(PART_HEX)} — передай objects (коллекцию или список)")
    colours = {o.name: PART_HEX[i] for i, o in enumerate(targets)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    r, d, sh, vs = scene.render, scene.display, scene.display.shading, scene.view_settings
    saved = {
        (scene, "camera"): scene.camera,
        (r, "engine"): r.engine, (r, "filepath"): r.filepath, (r, "film_transparent"): r.film_transparent,
        (r, "dither_intensity"): r.dither_intensity, (r, "use_compositing"): r.use_compositing,
        (r, "use_sequencer"): r.use_sequencer, (r, "resolution_x"): r.resolution_x, (r, "resolution_y"): r.resolution_y,
        (r, "resolution_percentage"): r.resolution_percentage,
        (r.image_settings, "file_format"): r.image_settings.file_format, (r.image_settings, "color_mode"): r.image_settings.color_mode,
        (r.image_settings, "color_depth"): r.image_settings.color_depth,
        (vs, "view_transform"): vs.view_transform, (vs, "look"): vs.look, (vs, "exposure"): vs.exposure, (vs, "gamma"): vs.gamma,
        (d, "render_aa"): d.render_aa,
        (sh, "light"): sh.light, (sh, "color_type"): sh.color_type, (sh, "show_cavity"): sh.show_cavity,
        (sh, "show_shadows"): sh.show_shadows, (sh, "show_object_outline"): sh.show_object_outline,
        (sh, "show_specular_highlight"): sh.show_specular_highlight, (sh, "show_xray"): sh.show_xray, (sh, "use_dof"): sh.use_dof,
    }
    for o in scene.objects:
        saved[(o, "color")] = tuple(o.color)
        saved[(o, "hide_render")] = o.hide_render
    try:
        scene.camera = cam
        r.engine = "BLENDER_WORKBENCH"
        r.film_transparent = True
        r.dither_intensity = 0
        r.use_compositing = False
        r.use_sequencer = False
        if resolution:
            r.resolution_x, r.resolution_y = resolution
            r.resolution_percentage = 100
        r.image_settings.file_format = "PNG"
        r.image_settings.color_mode = "RGBA"
        r.image_settings.color_depth = "8"
        vs.view_transform, vs.look, vs.exposure, vs.gamma = "Standard", "None", 0.0, 1.0
        d.render_aa = "OFF"  # без сглаживания: на краях не бывает смешанных цветов
        sh.light, sh.color_type = "FLAT", "OBJECT"
        sh.show_cavity = sh.show_shadows = sh.show_object_outline = sh.show_specular_highlight = sh.show_xray = sh.use_dof = False
        chosen = {o.name for o in targets}
        for o in scene.objects:
            if o.type in RENDERABLE and o.name not in chosen:
                o.hide_render = True
        for o in targets:
            o.color = _linear(colours[o.name])
        r.filepath = str(path)
        bpy.ops.render.render(write_still=True)
    finally:
        for (owner, attr), value in saved.items():
            try:
                setattr(owner, attr, value)
            except (AttributeError, TypeError, ValueError):
                pass

    (path.with_suffix(".json")).write_text(
        json.dumps({"camera": camera, "objects": colours}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # Проверка: сколько пикселей каждого цвета и нет ли пикселей «ничьего» цвета (значит, цвета исказились).
    img = bpy.data.images.load(str(path), check_existing=False)
    try:
        w, h = img.size
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
    finally:
        bpy.data.images.remove(img)
    px = (px.reshape(-1, 4) * 255).round().astype(np.int16)
    shown = px[px[:, 3] >= 128, :3]
    counts, matched = {}, np.zeros(len(shown), bool)
    for name, hx in colours.items():
        rgb = np.array([int(hx[i : i + 2], 16) for i in (1, 3, 5)], np.int16)
        hit = np.abs(shown - rgb).max(axis=1) <= CHECK_TOLERANCE
        counts[name] = int(hit.sum())
        matched |= hit
    unknown = float((~matched).mean()) if len(shown) else 0.0
    return {
        "png": str(path),
        "json": str(path.with_suffix(".json")),
        "size": [w, h],
        "pixels": {n: c for n, c in counts.items() if c},
        "not_visible": [n for n, c in counts.items() if not c],
        "unmatched_share": round(unknown, 4),
        "check": "ok" if unknown < 0.01 else "цвета на рендере не совпали с заданными — проверь управление цветом",
    }
