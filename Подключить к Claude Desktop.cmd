@echo off
chcp 65001 >nul
rem Подключает MCP-сервер «Зрение» к Claude Desktop. Запускать при закрытом Claude Desktop.
"%~dp0.venv\Scripts\python.exe" -m zrenie.install_desktop
pause
