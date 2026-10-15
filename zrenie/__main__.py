"""Запуск: python -m zrenie (stdio)."""

import logging
import sys

from .server import mcp


def main() -> None:
    # stdout занят протоколом MCP — логи только в stderr.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    mcp.run()


if __name__ == "__main__":
    main()
