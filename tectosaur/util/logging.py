import logging
import sys

def get_caller_logger():
    import sys
    module = None
    mod_name = sys._getframe(2).f_globals['__name__']
    return logging.getLogger(mod_name)

def setup_root_logger(log_name):
    L = logging.getLogger(log_name)
    L.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format)
    ch.setFormatter(formatter)
    L.addHandler(ch)
    return L
