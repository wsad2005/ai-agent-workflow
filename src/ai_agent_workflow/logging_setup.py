"""日志配置：只在应用入口调用；库与工具代码不配置 handler。"""

import logging


def configure_logging(verbose: bool = False) -> None:
    """配置根日志格式与级别；重复调用只调整级别，不叠加 handler。"""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)
