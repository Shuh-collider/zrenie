# Зрение (zrenie)

**MCP server that helps Claude see reference images precisely and build 3D models from them in Blender / Cinema 4D.** It splits turnaround sheets into views, removes backgrounds, aligns views to one scale, overlays grids, measures what Claude marks, extracts contours and colours, segments parts with SAM 2.1 and compares renders of the model with the references — for the whole silhouette and part by part. The server does not "understand" images: it prepares them and computes exact numbers, Claude looks and decides. Tool descriptions and output are in Russian.

---

MCP-сервер, через который Claude хорошо видит референсы (чаще всего сгенерированные ИИ картинки, 1–6 видов объекта) и строит по ним 3D-модели. Сервер готовит изображения и считает точные цифры, а смотрит и решает Claude.

Статус: рабочий прототип. Проверено на Windows 10, Python 3.11, Claude Code и Claude Desktop.

## Что умеет

| Инструмент | Что делает |
|---|---|
| `prepare_references` | начало работы одним вызовом: листы и файлы видов → виды без фона, выровненные, с несостыковками (+ экспорт) |
| `list_media` | картинки и видео в папке с размерами |
| `inspect` | обзор картинки; если это лист из нескольких видов — кадры обведены и пронумерованы |
| `split_views` | лист → каждый кадр крупно (кадры встык ищутся по швам, с промежутками — по промежуткам) |
| `view` | картинка или область крупно, режимы `normal` / `contrast` / `edges` / `tones` |
| `sheet` | несколько картинок на одном листе с подписями |
| `silhouette` | объект без фона: рамка, пропорции, части, отверстия, ключевые высоты |
| `align_views` | виды одной высоты на общей линии низа + несостыковки между видами |
| `export_views` | PNG без фона одной высоты + `views.json` для подложек в Blender / C4D |
| `grid` | сетка в пикселях оригинала или в процентах рамки объекта |
| `mark` | точки и линии Claude → длины, углы, доли объекта, привязка к краю |
| `contour` | контур и ось симметрии в координатах сцены, профиль для тела вращения, JSON для Blender |
| `palette` | основные цвета: HEX, sRGB и линейные значения для материалов |
| `compare` / `compare_views` | рендер модели против референса: совпадение силуэтов, где шире / уже, чего не хватает |
| `parts` | детали по точкам и рамкам (SAM 2.1): размер, место, наклон, цвет каждой |
| `compare_parts` | сравнение модели с референсом по деталям |

Вид задаётся строкой: `путь`, `путь|N` (кадр N листа), `путь|0` (вся картинка), `путь|x1,y1,x2,y2` (область); с деталью из `parts` — `путь|имя` или `путь|N|имя`. Все координаты — пиксели оригинала.

Размеры картинок подбираются под лимиты Claude на изображения: крупные виды (`view`, `grid`, `mark`, `parts`) — длинная сторона ≤ 2000 px и ≤ 4784 визуальных токенов, обзорные (`inspect`, `silhouette`, `contour`, `align_views`, сравнения) — до ~1600 токенов, чтобы разговор дольше не забивался картинками. Мелкие области увеличиваются, к каждой картинке приходит формула перевода координат, а на `inspect`, `split_views`, `view`, `mark` и `parts` по верхнему и левому краю — линейки с подписями в пикселях оригинала.

## Установка

Нужен Python 3.11+.

```powershell
python -m venv .venv

# Для нейросетей (BiRefNet — фон, SAM 2.1 — детали) нужен torch с CUDA. Ставить до остального:
.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu130

.venv\Scripts\python.exe -m pip install -e ".[ai,dev]"
```

Без видеокарты: `pip install -e ".[dev]"` — фон убирается по цвету (`method="simple"`), инструмент `parts` недоступен.

Веса нейросетей скачиваются с Hugging Face один раз и дальше берутся с диска: BiRefNet ~450 МБ, SAM 2.1 large ~0,9 ГБ. BiRefNet загружается в фоне сразу при запуске сервера, SAM 2.1 — при первом вызове `parts`.

## Подключение

**Claude Code** — файл `.mcp.json` в корне проекта (путь — к своему `.venv`):

```json
{
  "mcpServers": {
    "zrenie": {
      "command": "C:\\путь\\к\\zrenie\\.venv\\Scripts\\python.exe",
      "args": ["-m", "zrenie"]
    }
  }
}
```

**Claude Desktop (Windows)** — полностью закрыть Claude Desktop (значок в трее → Quit) и запустить `Подключить к Claude Desktop.cmd`. Установщик пишет в оба места, где Claude Desktop хранит настройки (в том числе версия из Microsoft Store), и делает резервную копию. На других системах — добавить тот же блок `zrenie` в `claude_desktop_config.json` вручную.

## Blender: карта деталей для `compare_parts`

`zrenie/blender_kit.py` запускается внутри Blender (например, через MCP Blender):

```python
exec(open(r"<папка проекта>\zrenie\blender_kit.py", encoding="utf-8").read())
result = render_parts("cam_front", r"C:\renders\front_parts.png")
```

Каждый объект сцены рендерится своим плоским цветом с прозрачным фоном, рядом пишется JSON «объект → цвет», настройки сцены потом возвращаются. По этой карте сервер вырезает детали модели без нейросети.

## Рабочий цикл

1. `prepare_references` — разобрать листы на виды, убрать фон, привести виды к одному масштабу и найти несостыковки генерации одним вызовом (по шагам — `inspect`, `split_views`, `align_views`, `export_views`).
2. Если что-то разрезалось не так — `inspect` и `align_views` с нужными кадрами `путь|N`.
3. `grid`, `mark`, `contour`, `palette`, `parts` — снять размеры, профили, цвета, детали.
4. Построить модель, отрендерить те же ракурсы ортографической камерой с прозрачным фоном.
5. `compare_views` и `compare_parts` — где модель расходится с референсом; поправить и повторить.

Лучше всего работает на референсах: каждый вид отдельным файлом от 1024 px, однотонный светло-серый фон без теней, ортографические ракурсы (спереди, сбоку, сзади, сверху + 3/4), один объект целиком с полями 5–10 %, без текста поверх.

## Тесты

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Тесты BiRefNet и SAM пропускаются, если нейросети не установлены или нет весов. `tests/refs/` — настоящие листы видов, рендеры и сгенерированный набор видов наушников.

## Лицензия

Код — [Apache License 2.0](LICENSE).

Нейросети в репозиторий не входят и скачиваются отдельно под своими лицензиями: [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) — MIT, [SAM 2.1](https://huggingface.co/facebook/sam2.1-hiera-large) — Apache 2.0.
