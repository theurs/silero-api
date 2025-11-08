import re
import razdel
from razdel import sentenize
import xml.etree.ElementTree as ET


MAX_CHUNK_LENGTH = 800


# --- КЛАСС ДЛЯ НАРЕЗКИ SSML ---
class SSMLSplitter:
    def __init__(self, max_len=MAX_CHUNK_LENGTH):
        self.max_len = max_len; self.chunks = []; self._reset_current_chunk()
    def _reset_current_chunk(self):
        self.current_root = ET.Element('speak'); self.current_char_count = 0
    def _finalize_chunk(self):
        if self.current_char_count > 0: self.chunks.append(ET.tostring(self.current_root, encoding='unicode'))
    def _rebuild_path(self, path):
        parent = self.current_root
        for tag, attribs in path: parent = ET.SubElement(parent, tag, attribs)
        return parent
    def _process_text(self, text, parent_element, path, is_tail=False):
        """
        Обрабатывает текстовый узел, разбивает на предложения и добавляет в текущий фрагмент,
        корректно обрабатывая пробелы между предложениями.
        """
        if not text or not text.strip():
            return

        for sentence in [s.text for s in sentenize(text)]:
            # Если добавление нового предложения превысит лимит, завершаем текущий фрагмент
            if self.current_char_count + len(sentence) > self.max_len and self.current_char_count > 0:
                self._finalize_chunk()
                self._reset_current_chunk()
                parent_element = self._rebuild_path(path)

            # Определяем, к какому элементу добавлять текст
            target_element = parent_element[-1] if is_tail and len(parent_element) > 0 else parent_element

            if is_tail:
                # Работаем с .tail (текст после закрывающего тега)
                current_tail = target_element.tail or ''
                # Добавляем пробел, если нужно
                if current_tail and not current_tail.endswith(' '):
                    current_tail += ' '
                target_element.tail = current_tail + sentence
            else:
                # Работаем с .text (текст внутри тега)
                current_text = target_element.text or ''
                # Добавляем пробел, если нужно
                if current_text and not current_text.endswith(' '):
                    current_text += ' '
                target_element.text = current_text + sentence

            # Обновляем счетчик символов с учетом возможного пробела
            self.current_char_count += len(sentence) + 1
    def _traverse(self, source_node, dest_parent, path):
        new_node = ET.SubElement(dest_parent, source_node.tag, source_node.attrib)
        current_path = path + [(source_node.tag, source_node.attrib)]
        self._process_text(source_node.text, new_node, current_path)
        for child in source_node:
            self._traverse(child, new_node, current_path)
            self._process_text(child.tail, new_node, current_path, is_tail=True)
    def split(self, ssml_string: str):
        self.chunks = []; self._reset_current_chunk()
        try:
            ssml_string = ssml_string.replace('xmlns="http://www.w3.org/2001/10/synthesis"', '')
            source_root = ET.fromstring(f"<speak>{ssml_string}</speak>" if not ssml_string.strip().startswith('<speak>') else ssml_string)
        except ET.ParseError as e: return [ssml_string]
        for child in source_root:
            self._traverse(child, self.current_root, [])
            self._process_text(child.tail, self.current_root, [], is_tail=True)
        self._finalize_chunk()
        return self.chunks


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