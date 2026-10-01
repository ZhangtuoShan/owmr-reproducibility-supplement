# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thur Nov 2023
@author: ayq
版权信息：华为技术有限公司，版本所有(C) 2021-2028
"""

import os
import logging
import datetime

from logging.handlers import RotatingFileHandler

def my_mkdir(path):
    path = os.getcwd() + path
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    return path
LOG_DIR = my_mkdir('/log')

def record_log():
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    LOG_DIR

    log_path = os.path.join(LOG_DIR, "logger_{}_{}.log".format(
        'reinforcelearn-inv-replenish', datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d')))

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] - %(module)s.%(funcName)s (%('
        'filename)s:%(lineno)d) - %(message)s')

    file_handler = RotatingFileHandler(
        filename=log_path, maxBytes=10 * 1024 * 1024, backupCount=30)
    file_handler.setFormatter(formatter)

    # 在代码运行主程序处，打印出日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = record_log()


def timer(method):
    def call(*args, **kwargs):
        time0 = datetime.datetime.now(tz=datetime.timezone.utc)
        r = method(*args, **kwargs)
        tm = datetime.datetime.now(tz=datetime.timezone.utc) - time0
        logger.info(f'func @{method.__name__} takes {tm.total_seconds()} (s)')
        return r

    return call
