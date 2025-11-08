import logging
from logging.handlers import RotatingFileHandler
import sys

def setup_logging():
    """
    Настраивает запись логов в файл с ротацией.
    """
    # Определяем формат сообщений
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    
    # Создаем обработчик, который будет писать в файл 'app.log'
    # Ротация: 1 файл размером до 5MB, хранится 5 старых копий.
    log_handler = RotatingFileHandler(
        "app.log", 
        maxBytes=5 * 1024 * 1024, # 5 мегабайт
        backupCount=5,
        encoding='utf-8'
    )
    
    # Применяем формат к обработчику
    log_handler.setFormatter(log_formatter)
    
    # Получаем корневой логгер
    root_logger = logging.getLogger()
    
    # Устанавливаем уровень логирования (например, INFO)
    root_logger.setLevel(logging.INFO)
    
    # Удаляем все существующие обработчики (например, консольный)
    # Это важно, чтобы избежать дублирования логов
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    # Добавляем наш новый файловый обработчик
    root_logger.addHandler(log_handler)

    # Дополнительно: перенаправляем необработанные исключения в лог
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger().critical("Необработанное исключение:", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception