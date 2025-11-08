import os
import time
import numpy as np
import torch
from scipy.io.wavfile import write as write_wav

# Важно: мы импортируем уже созданный экземпляр tts_model из нашего модуля
from app.tts_model import tts_model

# --- Параметры теста ---
TEST_TEXT = "Привет, мир! Это тестовый запуск модели Silero."
SPEAKER = 'xenia'
SAMPLE_RATE = 48000
OUTPUT_FILENAME = "test_output.wav"

def run_test():
    """
    Выполняет изолированную проверку модуля tts_model.
    """
    print("--- Запуск микро-теста для tts_model ---")

    # 1. Загрузка модели
    try:
        print("Шаг 1: Загрузка модели...")
        start_time = time.time()
        tts_model.load()
        duration = time.time() - start_time
        print(f"Модель успешно загружена за {duration:.2f} секунд.")
    except Exception as e:
        print(f"\n[ОШИБКА] Не удалось загрузить модель: {e}")
        return

    # 2. Синтез речи
    try:
        print("\nШаг 2: Синтез речи...")
        print(f"Текст: '{TEST_TEXT}'")
        start_time = time.time()
        audio_tensor = tts_model.process_text(
            text=TEST_TEXT,
            speaker=SPEAKER,
            sample_rate=SAMPLE_RATE,
            put_accent=True,
            put_yo=True
        )
        duration = time.time() - start_time
        
        if not isinstance(audio_tensor, torch.Tensor) or audio_tensor.numel() == 0:
            print("\n[ОШИБКА] Модель вернула пустой или некорректный результат (не тензор).")
            return
            
        print(f"Синтез речи успешно выполнен за {duration:.2f} секунд.")
        print(f"Размер полученного тензора: {audio_tensor.shape}")

    except Exception as e:
        print(f"\n[ОШИБКА] Произошла ошибка во время синтеза речи: {e}")
        return

    # 3. Сохранение в файл
    try:
        print(f"\nШаг 3: Сохранение аудио в файл '{OUTPUT_FILENAME}'...")
        # Конвертируем тензор PyTorch в массив NumPy (int16)
        audio_numpy = audio_tensor.numpy()
        audio_int16 = (audio_numpy * 32767).astype(np.int16)
        
        write_wav(OUTPUT_FILENAME, SAMPLE_RATE, audio_int16)
        
        # Получаем полный путь к файлу для удобства
        full_path = os.path.abspath(OUTPUT_FILENAME)
        print(f"Аудиофайл успешно сохранен: {full_path}")

    except Exception as e:
        print(f"\n[ОШИБКА] Не удалось сохранить аудиофайл: {e}")
        return

    print("\n--- Микро-тест успешно завершен! ---")
    print("Прослушайте файл 'test_output.wav', чтобы убедиться в корректности результата.")


if __name__ == "__main__":
    run_test()