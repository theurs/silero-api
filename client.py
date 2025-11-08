import os
import requests

# --- НАСТРОЙКИ ---

# URL вашего запущенного API
API_URL = "http://localhost:8000/tts/text"

# Текст, который вы хотите озвучить
TEXT_TO_SYNTHESIZE = """Силеро ТиТиЭс ФастАпэИ Сервис. Это готовый к работе, высокопроизводительный ФастАпэИ-сервис для синтеза русской речи с использованием моделей Силеро Моделс. Сервис предоставляет Рэст АпэИ для преобразования как обычного текста, так и текста с ЭсЭсЭмЭл-разметкой в аудиоформат Огэгэ Опус. Он спроектирован с учетом стабильности и производительности, включая обработку длинных текстов, потокобезопасность и централизованное логирование."""

# Путь для сохранения. ВНИМАНИЕ: Замените 'user' на ваше имя пользователя в Windows!
# Использование r'' (raw string) важно, чтобы Python правильно обработал обратные слэши.
OUTPUT_FOLDER = r"C:\Users\user\Downloads"

# Имя выходного файла
OUTPUT_FILENAME = "api_output.ogg"

# --- КОНЕЦ НАСТРОЕК ---


def main():
    """
    Основная функция для отправки запроса и сохранения файла.
    """
    print(f"[*] Отправка текста на синтез в API: {API_URL}")
    print(f"[*] Текст: \"{TEXT_TO_SYNTHESIZE[:50]}...\"")

    # 1. Формируем JSON-тело запроса
    payload = {
        "text": TEXT_TO_SYNTHESIZE,
        "speaker": "xenia",
        "put_accent": True,
        "put_yo": True
    }

    # 2. Выполняем POST-запрос
    try:
        response = requests.post(API_URL, json=payload, timeout=120) # Таймаут 120 секунд для длинных текстов

        # 3. Проверяем, что запрос прошел успешно (код ответа 200)
        if response.status_code == 200:
            
            # 4. Собираем полный путь для сохранения файла
            # Проверяем, существует ли папка, и создаем ее, если нужно
            if not os.path.exists(OUTPUT_FOLDER):
                print(f"[WARNING] Папка {OUTPUT_FOLDER} не найдена. Попытка создать...")
                os.makedirs(OUTPUT_FOLDER)

            full_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)

            # 5. Сохраняем полученные аудио-данные в файл
            # 'wb' означает запись в бинарном режиме (write binary), что критически важно для аудиофайлов
            with open(full_path, "wb") as audio_file:
                audio_file.write(response.content)
            
            print(f"\n[SUCCESS] Аудиофайл успешно сохранен!")
            print(f"[PATH]    {full_path}")

        else:
            # Если сервер вернул ошибку, выводим ее
            print(f"\n[ERROR] Сервер вернул ошибку: {response.status_code}")
            print(f"[DETAILS] {response.text}")

    except requests.exceptions.RequestException as e:
        # Если не удалось подключиться к серверу
        print(f"\n[CRITICAL] Не удалось подключиться к серверу.")
        print(f"[DETAILS]  Убедитесь, что сервис запущен по адресу {API_URL}")
        print(f"[ERROR]    {e}")


if __name__ == "__main__":
    
    main()