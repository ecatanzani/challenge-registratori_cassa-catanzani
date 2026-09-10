import logging


def setup_main_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up and configures the main application logger.

    This function simulates the logger definition being done in a 'main' part 
    of the application, ready to be passed to other modules/classes.
    """

    logger_name = name
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info(
        f"Logger '{logger_name}' initialized at level {logging.getLevelName(level)}")
    return logger