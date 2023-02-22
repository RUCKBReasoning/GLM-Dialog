# logger.py

import logging

logger = logging.getLogger('xkdai_logger')
logger.setLevel(logging.DEBUG)

# 创建一个文件处理器，将日志写入文件
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)

# 创建一个控制台处理器，将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 创建一个格式化程序，将日期、时间、级别和消息格式化
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 将格式化程序添加到处理器
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将处理器添加到Logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

