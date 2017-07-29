import logging
import sys

def setup_logger(log_name):
    L = logging.getLogger(log_name)
    L.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    L.addHandler(ch)
    return L
