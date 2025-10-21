import logging
import logging.handlers
import sys

def setup_logger():
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        return
    root_logger.setLevel(logging.INFO)

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)