import io
import subprocess
import sys
from contextlib import asynccontextmanager
import logging
import os
import uuid

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

# Импортируем нашу модель, локер и новый обработчик текста
from .tts_model import tts_model, tts_lock
from .text_processor import TextChunker, SSMLSplitter

# Получаем экземпляр логгера для текущего модуля
logger = logging.getLogger(__name__)


# --- Модели данных (Pydantic) для валидации запросов ---

class TextToSpeechRequest(BaseModel):
    # Убираем ограничение на максимальную длину, т.к. теперь обрабатываем длинные тексты
    text: str = Field(..., 
                      min_length=1, 
                      title="Текст для синтеза",
                      description="Текст, который нужно озвучить. Может быть длинным.")
    speaker: str = Field("xenia", 
                         title="Голос (диктор)",
                         description="Имя диктора из доступных в модели Silero v5.")
    put_accent: bool = Field(True, title="Расставлять ударения")
    put_yo: bool = Field(True, title="Расставлять букву ё")

class SsmlToSpeechRequest(BaseModel):
    ssml_text: str = Field(..., 
                           min_length=10, 
                           title="SSML разметка",
                           description="Текст в формате SSML. Максимум 2000 символов.")
    speaker: str = Field("xenia", 
                         title="Голос (диктор)",
                         description="Имя диктора из доступных в модели Silero v5.")


# --- Менеджер жизненного цикла для загрузки модели ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управляет жизненным циклом приложения.
    """
    logger.info("Приложение запускается...")
    tts_model.load()
    logger.info("Модель загружена, приложение готово к работе.")
    
    yield
    
    logger.info("Приложение останавливается.")


# --- Инициализация FastAPI ---

app = FastAPI(
    title="Silero TTS API",
    description="Простой API для синтеза русской речи с использованием Silero.",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешает запросы с любых сайтов
    allow_credentials=True,
    allow_methods=["*"],  # Разрешает все методы (POST, GET и т.д.)
    allow_headers=["*"],  # Разрешает все заголовки
)


# --- Вспомогательная функция для конвертации аудио ---

def convert_to_ogg_opus(audio_tensor: torch.Tensor, sample_rate: int, output_path: str) -> None:
    """
    Конвертирует аудио-тензор в файл формата Ogg Opus с помощью ffmpeg.
    """
    wav_in_memory = io.BytesIO()
    audio_numpy = audio_tensor.numpy()
    audio_int16 = (audio_numpy * 32767).astype(np.int16)
    write_wav(wav_in_memory, sample_rate, audio_int16)
    wav_in_memory.seek(0)

    try:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
                "-f", "opus", "-b:a", "24k",
                output_path
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        _, stderr = process.communicate(input=wav_in_memory.read())

        if process.returncode != 0:
            error_message = f"Ошибка ffmpeg: {stderr.decode()}"
            logger.error(error_message)
            raise RuntimeError(error_message)

    except FileNotFoundError:
        error_message = "Ошибка: ffmpeg не найден. Убедитесь, что он установлен и доступен в системном PATH."
        logger.critical(error_message)
        raise RuntimeError(error_message)


# --- Эндпоинты API ---

@app.post(
    "/tts/text",
    tags=["Синтез речи"],
    summary="Синтезировать речь из обычного текста",
    response_description="Аудиофайл в формате Ogg Opus.",
    responses={
        200: { "content": {"audio/ogg": {}}, "description": "Успешный синтез речи." },
        400: { "description": "Некорректный текст для синтеза." },
        500: { "description": "Внутренняя ошибка сервера." }
    }
)
def text_to_speech_endpoint(request: TextToSpeechRequest):
    SAMPLE_RATE = 48000
    try:
        # 1. Нарезаем текст на безопасные фрагменты
        chunker = TextChunker()
        text_chunks = chunker.split(request.text)
        
        if not text_chunks:
            raise HTTPException(status_code=400, detail="Текст пуст или не содержит поддерживаемых символов после очистки.")

        all_audio_tensors = []
        # 2. Синтезируем каждый фрагмент в цикле
        with tts_lock:
            for i, chunk in enumerate(text_chunks):
                logger.info(f"Синтез фрагмента {i+1}/{len(text_chunks)}: '{chunk[:50]}...'")
                audio_tensor = tts_model.process_text(
                    text=chunk,
                    speaker=request.speaker,
                    sample_rate=SAMPLE_RATE,
                    put_accent=request.put_accent,
                    put_yo=request.put_yo
                )
                all_audio_tensors.append(audio_tensor)
        
        # 3. Склеиваем все аудио-фрагменты
        if not all_audio_tensors:
            raise HTTPException(status_code=500, detail="Не удалось сгенерировать аудио ни для одного из фрагментов текста.")
        final_audio = torch.cat(all_audio_tensors)

        # 4. Сохраняем и возвращаем финальный файл
        temp_file_path = f"{uuid.uuid4().hex}.ogg"
        convert_to_ogg_opus(final_audio, SAMPLE_RATE, temp_file_path)
        
        cleanup_task = BackgroundTask(os.remove, path=temp_file_path)
        return FileResponse(
            path=temp_file_path, 
            media_type="audio/ogg", 
            filename="speech.ogg",
            background=cleanup_task
        )

    except ValueError:
        logger.warning(f"ValueError от Silero для одного из фрагментов текста.")
        raise HTTPException(status_code=400, detail="Один из фрагментов текста не может быть обработан моделью.")
    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /tts/text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/tts/ssml",
    tags=["Синтез речи"],
    summary="Синтезировать речь из SSML",
    response_description="Аудиофайл в формате Ogg Opus.",
    responses={
        200: { "content": {"audio/ogg": {}}, "description": "Успешный синтез речи." },
        400: { "description": "Некорректный SSML для синтеза." },
        500: { "description": "Внутренняя ошибка сервера." }
    }
)

def ssml_to_speech_endpoint(request: SsmlToSpeechRequest):
    SAMPLE_RATE = 48000
    try:
        # 1. Нарезаем SSML на безопасные фрагменты
        splitter = SSMLSplitter()
        ssml_chunks = splitter.split(request.ssml_text)

        if not ssml_chunks:
            raise HTTPException(status_code=400, detail="SSML пуст или некорректен.")

        all_audio_tensors = []
        # 2. Синтезируем каждый фрагмент в цикле
        with tts_lock:
            for i, chunk in enumerate(ssml_chunks):
                logger.info(f"Синтез SSML фрагмента {i+1}/{len(ssml_chunks)}: '{chunk[:70]}...'")
                audio_tensor = tts_model.process_ssml(
                    ssml_text=chunk,
                    speaker=request.speaker,
                    sample_rate=SAMPLE_RATE
                )
                all_audio_tensors.append(audio_tensor)
        
        # 3. Склеиваем все аудио-фрагменты
        if not all_audio_tensors:
            raise HTTPException(status_code=500, detail="Не удалось сгенерировать аудио ни для одного из SSML-фрагментов.")
        final_audio = torch.cat(all_audio_tensors)

        # 4. Сохраняем и возвращаем финальный файл
        temp_file_path = f"{uuid.uuid4().hex}.ogg"
        convert_to_ogg_opus(final_audio, SAMPLE_RATE, temp_file_path)
        
        cleanup_task = BackgroundTask(os.remove, path=temp_file_path)
        return FileResponse(
            path=temp_file_path, 
            media_type="audio/ogg", 
            filename="speech_ssml.ogg",
            background=cleanup_task
        )

    except ValueError:
        logger.warning(f"ValueError от Silero для одного из SSML-фрагментов.")
        raise HTTPException(status_code=400, detail="Один из SSML-фрагментов не может быть обработан моделью.")
    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /tts/ssml: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))