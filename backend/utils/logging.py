import logging
from pythonjsonlogger import jsonlogger

def setup_logging():
    logger = logging.getLogger("forensiclens")
    logger.setLevel(logging.INFO)
    
    
    if logger.hasHandlers():
        logger.handlers.clear()

    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(case_id)s %(file_hash)s %(message)s',
        rename_fields={"asctime": "timestamp"}
    )
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)
    return logger

logger = logging.getLogger("forensiclens")
