"""Подключает «Зрение» к Claude Desktop: добавляет сервер в claude_desktop_config.json.

Пока Claude Desktop открыт, он перезаписывает этот файл своей копией настроек, поэтому запускать —
только при закрытом Claude Desktop (через «Подключить к Claude Desktop.cmd» в папке проекта).
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SERVER_NAME = "zrenie"


def claude_running() -> bool:
    result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq claude.exe", "/NH"], capture_output=True, text=True)
    return "claude.exe" in result.stdout.lower()


def config_paths() -> list[Path]:
    """Где Claude Desktop держит настройки.

    Версия из Microsoft Store читает свою копию в AppData\\Local\\Packages\\Claude_*\\LocalCache\\Roaming\\Claude,
    а не AppData\\Roaming\\Claude. Пишем в обе — какую версию ни поставили, она увидит сервер.
    """
    paths = sorted(Path(os.environ["LOCALAPPDATA"], "Packages").glob("Claude_*/LocalCache/Roaming/Claude/claude_desktop_config.json"))
    paths.append(Path(os.environ["APPDATA"]) / "Claude" / "claude_desktop_config.json")
    return paths


def add_server(path: Path) -> list[str]:
    config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if path.exists():
        backup = path.with_name(f"claude_desktop_config.backup-{datetime.now():%Y%m%d-%H%M%S}.json")
        shutil.copy2(path, backup)
        print(f"Резервная копия: {backup}")
    servers = config.setdefault("mcpServers", {})
    servers[SERVER_NAME] = {"command": sys.executable, "args": ["-m", "zrenie"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return list(servers)


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # консоль Windows может не уметь в некоторые символы
    if claude_running():
        print("Claude Desktop ещё открыт. Закрой его полностью: значок в трее у часов, пункт Quit. Потом запусти этот файл снова.")
        return 1
    for path in config_paths():
        servers = add_server(path)
        print(f"Записано: {path}\n  серверы: {', '.join(servers)}")
    print("Готово: «Зрение» подключено. Теперь запусти Claude Desktop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
