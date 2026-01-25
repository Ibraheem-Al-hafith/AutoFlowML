import logging
import sys
from typing import List

class StreamlitHandler(logging.Handler):
    """Custom logging handler to store logs for Streamlit display."""
    def __init__(self):
        super().__init__()
        self.logs: List[str] = []

    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)

def get_logger(name: str = "AutoFlowML") -> logging.Logger:
    """Initializes a professional logger with console and buffer handlers."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console Handler for terminal output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Custom Streamlit Handler
        st_handler = StreamlitHandler()
        st_handler.setFormatter(formatter)
        logger.addHandler(st_handler)

    return logger

# Global instance for the app
logger = get_logger()