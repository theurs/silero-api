import re
import razdel

class TextChunker:
    """
    Класс для очистки и нарезки текста на фрагменты,
    пригодные для синтеза моделью Silero.
    """
    def __init__(self, max_chunk_length: int = 950):
        self.max_length = max_chunk_length

    def _sanitize_text(self, text: str) -> str:
        """
        Удаляет неподдерживаемые символы и нормализует пробелы.
        """
        # Whitelist символов: русские буквы, цифры, основные знаки препинания и пробелы
        allowed_chars_pattern = r'[^а-яА-ЯёЁ0-9\s.,!?-—:;]'
        text = re.sub(allowed_chars_pattern, ' ', text)
        text = re.sub(r'[ \t]+', ' ', text).strip()
        return text

    def _chunk_by_words(self, long_sentence: str) -> list[str]:
        """
        Аварийная нарезка слишком длинного предложения по словам.
        """
        words = list(razdel.tokenize(long_sentence))
        chunks = []
        current_chunk = ""
        for word in words:
            if len(current_chunk) + len(word.text) + 1 <= self.max_length:
                current_chunk += word.text + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = word.text + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def split(self, text: str) -> list[str]:
        """
        Основной метод, который нарезает текст на фрагменты.
        """
        # --- НОВАЯ УЛУЧШЕННАЯ ЛОГИКА ---
        
        # 1. Сначала применяем базовую очистку от невалидных символов
        text = self._sanitize_text(text)
        
        # 2. Разделяем текст на абзацы по двум или более переносам строк
        paragraphs = re.split(r'\n{2,}', text)
        
        cleaned_paragraphs = []
        for p in paragraphs:
            # 3. Внутри каждого абзаца заменяем одиночные переносы на пробел и убираем лишние пробелы по краям
            cleaned_p = re.sub(r'\n', ' ', p).strip()
            if cleaned_p:
                cleaned_paragraphs.append(cleaned_p)
                
        # 4. Соединяем очищенные абзацы через точку, чтобы гарантировать паузу
        processed_text = ". ".join(cleaned_paragraphs)

        # --- КОНЕЦ НОВОЙ ЛОГИКИ ---
        
        if not processed_text:
            return []

        final_chunks = []
        sentences = list(razdel.sentenize(processed_text))
        current_chunk = ""
        for sentence in sentences:
            if len(sentence.text) > self.max_length:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                    current_chunk = ""
                final_chunks.extend(self._chunk_by_words(sentence.text))
                continue

            if len(current_chunk) + len(sentence.text) + 1 <= self.max_length:
                current_chunk += sentence.text + " "
            else:
                final_chunks.append(current_chunk.strip())
                current_chunk = sentence.text + " "
        
        if current_chunk:
            final_chunks.append(current_chunk.strip())

        return [chunk for chunk in final_chunks if chunk]


# --- Тестовый блок для прямой проверки этого модуля ---
if __name__ == "__main__":
    chunker = TextChunker()
    # Текст, который вы хотите озвучить
    TEXT_TO_SYNTHESIZE = """Силеро ТиТиЭс ФастАпэИ Сервис



Это готовый к работе, высокопроизводительный ФастАпэИ-сервис для синтеза русской речи с использованием моделей Силеро Моделс.



Сервис предоставляет Рэст АпэИ для преобразования как обычного текста, так и текста с ЭсЭсЭмЭл-разметкой в аудиоформат Огэгэ Опус. Он спроектирован с учетом стабильности и производительности, включая обработку длинных текстов, потокобезопасность и централизованное логирование."""

    print("--- Исходный текст ---")
    print(TEXT_TO_SYNTHESIZE)
    print("\n--- Результат нарезки ---")
    chunks = chunker.split(TEXT_TO_SYNTHESIZE)
    for i, chunk in enumerate(chunks):
        print(f"Фрагмент {i+1} (длина: {len(chunk)}): \"{chunk}\"")