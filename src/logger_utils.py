import logging
import sys
import os

def setup_logger(log_path="workflow.log", verbose=False):
    """
    Sets up a logger that outputs to both a file and the console.
    """
    logger = logging.getLogger("AutoFlow-SRXN")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Avoid duplicate handlers if setup multiple times
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File Handler
    try:
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not setup file logging at {log_path}: {e}")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def get_workflow_logger():
    return logging.getLogger("AutoFlow-SRXN")
