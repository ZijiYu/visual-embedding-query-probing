import logging
import os

def setup_logger(log_file="/workspace/probing/Probing/logs/train_probe.log"):
    """
    创建一个同时输出到终端和文件的 logger。
    """

    logger = logging.getLogger("probe_logger")
    logger.setLevel(logging.INFO)

    # 若已添加 handler，避免重复打印
    if logger.handlers:
        return logger

    # formatter：包括时间、日志等级、消息内容
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1) 终端输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 2) 文件输出
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
