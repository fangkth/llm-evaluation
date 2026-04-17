"""日志配置工具。"""

import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    """配置全局日志，使用 Rich 美化输出。"""
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的 logger。"""
    return logging.getLogger(name)
