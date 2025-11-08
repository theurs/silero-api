import torch
import threading
import logging

# Потокобезопасная блокировка для доступа к модели
tts_lock = threading.Lock()

# Получаем логгер для этого модуля
logger = logging.getLogger(__name__)

class SileroTTS:
    """
    Класс-обертка для управления моделью Silero TTS.
    Обеспечивает загрузку модели и логирование запросов.
    """
    def __init__(self):
        self.model = None
        self.device = None
        logger.info("Экземпляр SileroTTS создан.")

    def load(self, language='ru', model_id='v5_ru', device_str='cpu'):
        """
        Загружает модель в память.
        """
        try:
            logger.info(f"Начало загрузки модели: язык={language}, модель={model_id}...")
            self.device = torch.device(device_str)
            
            model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                      model='silero_tts',
                                      language=language,
                                      speaker=model_id)
            
            model.to(self.device)
            self.model = model
            logger.info("Модель Silero успешно загружена и готова к работе.")
        except Exception as e:
            logger.critical(f"Критическая ошибка при загрузке модели: {e}", exc_info=True)
            raise

    def process_text(self, text: str, speaker: str, sample_rate: int, **kwargs) -> torch.Tensor:
        """
        Синтезирует речь из обычного текста.
        """
        if not self.model:
            raise RuntimeError("Модель не загружена. Вызовите метод .load() перед использованием.")

        # logger.info(f"Processing text (len: {len(text)}): '{text[:70]}...' | speaker='{speaker}', sample_rate={sample_rate}, options={kwargs}")

        return self.model.apply_tts(text=text,
                                    speaker=speaker,
                                    sample_rate=sample_rate,
                                    **kwargs)

    def process_ssml(self, ssml_text: str, speaker: str, sample_rate: int, **kwargs) -> torch.Tensor:
        """
        Синтезирует речь из SSML-текста.
        """
        if not self.model:
            raise RuntimeError("Модель не загружена. Вызовите метод .load() перед использованием.")
            
        # logger.info(f"Processing SSML (len: {len(ssml_text)}): '{ssml_text[:90]}...' | speaker='{speaker}', sample_rate={sample_rate}, options={kwargs}")
        
        return self.model.apply_tts(ssml_text=ssml_text,
                                    speaker=speaker,
                                    sample_rate=sample_rate,
                                    **kwargs)

# --- Глобальный экземпляр класса, который будет использоваться во всем приложении ---
tts_model = SileroTTS()