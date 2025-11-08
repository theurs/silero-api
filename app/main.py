import io
import subprocess
import logging
import os
import uuid

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from contextlib import asynccontextmanager

# Разрешаем CORS-запросы для веб-интерфейса
from fastapi.middleware.cors import CORSMiddleware

# Импортируем нашу модель, локер и обработчики текста/ssml
from .tts_model import tts_model, tts_lock
from .text_processor import TextChunker, SSMLSplitter

# Получаем экземпляр логгера для текущего модуля
logger = logging.getLogger(__name__)


# --- Модели данных (Pydantic) для валидации запросов ---

class TextToSpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, title="Текст для синтеза")
    speaker: str = Field("xenia", title="Голос (диктор)")
    put_accent: bool = Field(True, title="Расставлять ударения")
    put_yo: bool = Field(True, title="Расставлять букву ё")
    put_stress_homo: bool = Field(True, title="Ударения в омографах")
    put_yo_homo: bool = Field(True, title="Буква ё в омографах")

class SsmlToSpeechRequest(BaseModel):
    ssml_text: str = Field(..., min_length=10, title="SSML разметка")
    speaker: str = Field("xenia", title="Голос (диктор)")
    put_accent: bool = Field(True, title="Расставлять ударения")
    put_yo: bool = Field(True, title="Расставлять букву ё")
    put_stress_homo: bool = Field(True, title="Ударения в омографах")
    put_yo_homo: bool = Field(True, title="Буква ё в омографах")


# --- Менеджер жизненного цикла для загрузки модели ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Приложение запускается...")
    tts_model.load()
    logger.info("Модель загружена, приложение готово к работе.")
    yield
    logger.info("Приложение останавливается.")


# --- Инициализация FastAPI ---

app = FastAPI(
    title="Silero TTS API",
    description="API для синтеза русской речи с использованием Silero.",
    version="1.1.0",
    lifespan=lifespan
)

# --- Настройка CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Вспомогательная функция для конвертации аудио ---

def convert_to_ogg_opus(audio_tensor: torch.Tensor, sample_rate: int, output_path: str) -> None:
    wav_in_memory = io.BytesIO()
    audio_numpy = audio_tensor.numpy()
    audio_int16 = (audio_numpy * 32767).astype(np.int16)
    write_wav(wav_in_memory, sample_rate, audio_int16)
    wav_in_memory.seek(0)
    try:
        process = subprocess.Popen(
            ["ffmpeg", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
             "-f", "opus", "-b:a", "24k", output_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        _, stderr = process.communicate(input=wav_in_memory.read())
        if process.returncode != 0:
            error_message = f"Ошибка ffmpeg: {stderr.decode()}"
            logger.error(error_message)
            raise RuntimeError(error_message)
    except FileNotFoundError:
        error_message = "Ошибка: ffmpeg не найден. Убедитесь, что он установлен."
        logger.critical(error_message)
        raise RuntimeError(error_message)


# --- Эндпоинты API ---

@app.post("/tts/text", tags=["Синтез речи"], summary="Синтезировать речь из обычного текста")
def text_to_speech_endpoint(request: TextToSpeechRequest):
    SAMPLE_RATE = 48000
    try:
        chunker = TextChunker()
        text_chunks = chunker.split(request.text)
        if not text_chunks:
            raise HTTPException(status_code=400, detail="Текст пуст или некорректен.")

        all_audio_tensors = []
        with tts_lock:
            for i, chunk in enumerate(text_chunks):
                audio_tensor = tts_model.process_text(
                    text=chunk,
                    speaker=request.speaker,
                    sample_rate=SAMPLE_RATE,
                    put_accent=request.put_accent,
                    put_yo=request.put_yo,
                    put_stress_homo=request.put_stress_homo,
                    put_yo_homo=request.put_yo_homo
                )
                all_audio_tensors.append(audio_tensor)
        
        if not all_audio_tensors:
            raise HTTPException(status_code=500, detail="Не удалось сгенерировать аудио.")
        
        final_audio = torch.cat(all_audio_tensors)
        temp_file_path = f"{uuid.uuid4().hex}.ogg"
        convert_to_ogg_opus(final_audio, SAMPLE_RATE, temp_file_path)
        
        cleanup_task = BackgroundTask(os.remove, path=temp_file_path)
        return FileResponse(
            path=temp_file_path, media_type="audio/ogg", 
            filename="speech.ogg", background=cleanup_task
        )
    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /tts/text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/ssml", tags=["Синтез речи"], summary="Синтезировать речь из SSML")
def ssml_to_speech_endpoint(request: SsmlToSpeechRequest):
    SAMPLE_RATE = 48000
    try:
        splitter = SSMLSplitter()
        ssml_chunks = splitter.split(request.ssml_text)
        if not ssml_chunks:
            raise HTTPException(status_code=400, detail="SSML пуст или некорректен.")

        all_audio_tensors = []
        with tts_lock:
            for i, chunk in enumerate(ssml_chunks):
                audio_tensor = tts_model.process_ssml(
                    ssml_text=chunk,
                    speaker=request.speaker,
                    sample_rate=SAMPLE_RATE,
                    put_accent=request.put_accent,
                    put_yo=request.put_yo,
                    put_stress_homo=request.put_stress_homo,
                    put_yo_homo=request.put_yo_homo
                )
                all_audio_tensors.append(audio_tensor)
        
        if not all_audio_tensors:
            raise HTTPException(status_code=500, detail="Не удалось сгенерировать аудио.")

        final_audio = torch.cat(all_audio_tensors)
        temp_file_path = f"{uuid.uuid4().hex}.ogg"
        convert_to_ogg_opus(final_audio, SAMPLE_RATE, temp_file_path)
        
        cleanup_task = BackgroundTask(os.remove, path=temp_file_path)
        return FileResponse(
            path=temp_file_path, media_type="audio/ogg",
            filename="speech_ssml.ogg", background=cleanup_task
        )
    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /tts/ssml: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))