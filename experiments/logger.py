import logging

FORMATTER = logging.Formatter(
    "[%(asctime)-19.19s %(levelname)-1.1s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

MEMORY_HANDLER = logging.handlers.MemoryHandler(capacity=100)
MEMORY_HANDLER.setFormatter(FORMATTER)

LOGGER = logging.getLogger("main")
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(MEMORY_HANDLER)
LOGGING_INITIALIZED = False


class RankFilter(logging.Filter):
    def __init__(self, is_master):
        super().__init__()
        self.is_master = is_master

    def filter(self, record):
        return self.is_master
