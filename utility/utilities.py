# -*- coding: UTF-8 -*-

import datetime
import io
import logging
import os
import sys
import time

def get_logger(filepath):
    """
        Gets a logger instance to write the program info and errors to.
        @params:
            filepath (string): File path to the log output.
        @returns:
            Instance of a logger.
    """

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    handler = logging.FileHandler(filepath)
    handler.setFormatter(logging.Formatter(
        "%(levelname)s:%(message)s"
    ))

    logging.getLogger().addHandler(handler)

    with io.open(filepath, "a", encoding="utf-8") as lf:
        lf.write("\n=========================================================================\n")
        lf.write(get_timestamp() + "\n")
        lf.write("=========================================================================\n")

    return logger


def get_timestamp(pad=False):
    """
        Gets the date and timestamp in yyyy-MM-dd hh:mm:ss format for the current time
        using the datetime library.
        @params:
            pad (bool): Optional flag to determine if the string should be padded with brackets.
        @returns:
            String representation of the timestamp.
    """
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return "[" + time_str + "]" if pad else time_str