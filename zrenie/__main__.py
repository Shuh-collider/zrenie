"""Запуск: python -m zrenie (stdio)."""

import logging
import sys
import threading

from . import matting
from .server import mcp


def main() -> None:
    # stdout занят протоколом MCP — логи только в stderr.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # Нейросеть для фона грузится в фоне, пока Claude смотрит первые картинки, — первый silhouette не ждёт.
    threading.Thread(target=matting.warm_up, name="warm-up", daemon=True).start()
    mcp.run()


if __name__ == "__main__":
    main()
