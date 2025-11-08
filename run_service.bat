@echo off
:: Устанавливаем заголовок окна консоли
title Silero TTS Service

:: =================================================================
:: Этот бат-файл для запуска FastAPI сервиса Silero TTS.
:: Он автоматически активирует виртуальное окружение и запускает сервер.
:: Поместите этот файл в корневую папку проекта.
:: =================================================================

echo [INFO] Preparing to start the Silero TTS Service...

:: Переходим в директорию, где находится сам .bat файл.
:: Это гарантирует, что все относительные пути будут работать правильно.
cd /d %~dp0

:: Активируем виртуальное окружение
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

:: Запускаем Uvicorn сервер
echo [INFO] Launching Uvicorn server...
echo [INFO] The service will be available at http://localhost:8000
echo [INFO] Press CTRL+C in this window to stop the server.
echo.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-config log_config.yaml

echo.
echo [INFO] Server has been stopped.
pause
