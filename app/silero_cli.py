# silero_cli.py (финальная версия с нарезкой и без дребезжания)
# pyinstaller --onefile --name silero_cli silero_cli.py

import os
import argparse
import torch
import numpy as np
import subprocess
import tempfile
from scipy.io.wavfile import write as write_wav
from razdel import sentenize
import sys
import platform
import requests
from tqdm import tqdm
import zipfile
import tarfile
import xml.etree.ElementTree as ET

# --- НАСТРОЙКИ ---
MAX_CHUNK_LENGTH = 1000  # Увеличим лимит до более реалистичного
FFMPEG_EXE_NAME = 'ffmpeg.exe' if sys.platform == 'win32' else 'ffmpeg'
MODEL_ID = 'v5_ru' # Используем актуальную модель v5

# --- КЛАСС ДЛЯ НАРЕЗКИ SSML (из предыдущей рабочей версии) ---
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
        if not text or not text.strip(): return
        for sentence in [s.text for s in sentenize(text)]:
            if self.current_char_count + len(sentence) > self.max_len and self.current_char_count > 0:
                self._finalize_chunk(); self._reset_current_chunk(); parent_element = self._rebuild_path(path)
            target_element = parent_element[-1] if is_tail and len(parent_element) > 0 else parent_element
            if is_tail: target_element.tail = (target_element.tail or '') + sentence
            else: target_element.text = (target_element.text or '') + sentence
            self.current_char_count += len(sentence)
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

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def find_ffmpeg_path():
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), FFMPEG_EXE_NAME)
    if os.path.exists(local_path): return local_path
    try:
        subprocess.run([FFMPEG_EXE_NAME, '-version'], capture_output=True, check=True, text=True)
        return FFMPEG_EXE_NAME
    except (FileNotFoundError, subprocess.CalledProcessError): return None
def download_ffmpeg():
    system, machine = sys.platform, platform.machine()
    if system == 'win32': url, arc, exe = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip", "ffmpeg.zip", "ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
    elif system == 'linux': url, arc, exe = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz", "ffmpeg.tar.xz", "ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"
    elif system == 'darwin': url, arc, exe = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-osx64-gpl.zip", "ffmpeg.zip", "ffmpeg-master-latest-osx64-gpl/bin/ffmpeg"
    else: print(f"[ERROR] Автозагрузка ffmpeg для вашей системы ({system}/{machine}) не поддерживается."); return None
    if input(f"[*] FFmpeg не найден. Скачать его (~80MB)? (y/n): ").lower()!='y': return None
    try:
        resp = requests.get(url, stream=True); resp.raise_for_status()
        total_size = int(resp.headers.get('content-length', 0))
        with open(arc, 'wb') as f, tqdm(desc=arc, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
            for chunk in resp.iter_content(chunk_size=8192): f.write(chunk); bar.update(len(chunk))
        print(f"[*] Распаковка...");
        if arc.endswith(".zip"):
            with zipfile.ZipFile(arc, 'r') as zf: zf.extract(exe)
        else:
            with tarfile.open(arc, 'r:xz') as tf: tf.extract(exe)
        os.rename(exe, FFMPEG_EXE_NAME); os.remove(arc); os.rmdir(os.path.dirname(exe))
        print(f"[+] FFmpeg успешно скачан."); return find_ffmpeg_path()
    except Exception as e: print(f"[ERROR] Ошибка скачивания ffmpeg: {e}"); return None
def ensure_ffmpeg(): return find_ffmpeg_path() or download_ffmpeg()
def download_model():
    print(f"[*] Загрузка модели Silero ({MODEL_ID})...");
    try:
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='ru', speaker=MODEL_ID)
        model.to(torch.device('cpu')); print("[+] Модель успешно загружена."); return model
    except Exception as e: print(f"[ERROR] Не удалось загрузить модель: {e}"); return None
def split_text_into_chunks(text: str):
    chunks, current_chunk = [], ""
    for sentence in [s.text for s in sentenize(text)]:
        if len(current_chunk) + len(sentence) < MAX_CHUNK_LENGTH: current_chunk += " " + sentence
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks
def save_as_wav(audio_tensor, path, sample_rate):
    audio_int16 = (audio_tensor.numpy() * 32767).astype(np.int16)
    write_wav(path, sample_rate, audio_int16)
def convert_wav_to_ogg(wav_path, ogg_path, ffmpeg_path):
    print("[*] Конвертация в OGG...")
    command = [ffmpeg_path, '-i', wav_path, '-acodec', 'libopus', '-b:a', '24k', '-vbr', 'on', '-y', '-loglevel', 'error', ogg_path]
    try: subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as e: print(f"[ERROR] Ошибка конвертации в OGG: {e.stderr}"); return False
    return True

# --- ОСНОВНАЯ ЛОГИКА ---
def main():
    parser = argparse.ArgumentParser(description="Консольная утилита для синтеза речи Silero TTS.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Текст для синтеза."); group.add_argument("--file", type=str, help="Путь к текстовому файлу.")
    group.add_argument("--ssml", type=str, help="Путь к файлу с SSML-разметкой.")
    parser.add_argument("--save", type=str, required=True, help="Путь для сохранения аудио (.wav или .ogg).")
    parser.add_argument("--speaker", type=str, default="xenia", help="Голос (aidar, baya, kseniya, xenia, eugene, random).")
    parser.add_argument("--sample_rate", type=int, default=48000, choices=[8000, 24000, 48000], help="Частота дискретизации.")
    args = parser.parse_args()

    try:
        if args.ssml:
            with open(args.ssml, 'r', encoding='utf-8') as f: input_data = f.read()
            is_ssml = True; print(f"[*] Чтение SSML из файла: {args.ssml}")
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f: input_data = f.read()
            is_ssml = False; print(f"[*] Чтение текста из файла: {args.file}")
        else: input_data = args.text; is_ssml = False; print("[*] Используется текст из командной строки.")
    except FileNotFoundError as e: print(f"[ERROR] Файл не найден: {e.filename}"); return
    
    model = download_model();
    if not model: return

    if is_ssml: chunks = SSMLSplitter().split(input_data)
    else: chunks = split_text_into_chunks(input_data)
    if not chunks: print("[ERROR] Не удалось получить фрагменты для синтеза."); return
    
    audio_tensors = []
    # Создаем тензор тишины для плавной склейки
    silence_duration_ms = 250
    silence_tensor = torch.zeros(int(args.sample_rate * silence_duration_ms / 1000))

    print(f"[*] Входные данные разбиты на {len(chunks)} фрагментов. Начинаю синтез...")
    for i, chunk in enumerate(chunks):
        print(f"    - Синтез фрагмента {i+1}/{len(chunks)}...")
        try:
            params = {'speaker': args.speaker, 'sample_rate': args.sample_rate}
            if is_ssml: params['ssml_text'] = chunk
            else: params.update({'text': chunk, 'put_accent': True, 'put_yo': True, 'put_stress_homo': True, 'put_yo_homo': True})
            
            audio = model.apply_tts(**params)
            audio_tensors.append(audio)
            if i < len(chunks) - 1: audio_tensors.append(silence_tensor) # Добавляем тишину между фрагментами
        except Exception as e: print(f"[WARNING] Ошибка при синтезе фрагмента {i+1}: {e}")
    
    if not audio_tensors: print("[ERROR] Не удалось синтезировать ни одного аудиофрагмента."); return
    
    final_audio = torch.cat(audio_tensors)
    print("[+] Синтез успешно завершен.")

    output_path = args.save; file_ext = os.path.splitext(output_path)[1].lower()
    if file_ext == '.ogg':
        ffmpeg_exe = ensure_ffmpeg()
        if not ffmpeg_exe: print("[ERROR] Невозможно сохранить в .ogg без ffmpeg."); return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            save_as_wav(final_audio, tmp.name, args.sample_rate)
            if convert_wav_to_ogg(tmp.name, output_path, ffmpeg_exe):
                print(f"\n[SUCCESS] Аудиофайл успешно сохранен: {os.path.abspath(output_path)}")
        os.remove(tmp.name)
    elif file_ext == '.wav':
        save_as_wav(final_audio, output_path, args.sample_rate)
        print(f"\n[SUCCESS] Аудиофайл успешно сохранен: {os.path.abspath(output_path)}")
    else: print(f"[ERROR] Неподдерживаемый формат: '{file_ext}'. Используйте .wav или .ogg.")

if __name__ == "__main__":
    main()