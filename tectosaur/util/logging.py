import logging
import sys

def get_caller_logger():
    import sys
    module = None
    mod_name = sys._getframe(2).f_globals['__name__']
    return logging.getLogger(mod_name)

loggers = {}

def setup_logger(log_name):
    global loggers
    if log_name in loggers:
        return loggers[log_name]

    L = logging.getLogger(log_name)
    L.setLevel(logging.DEBUG)
    L.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format)
    ch.setFormatter(formatter)
    L.addHandler(ch)

    loggers[log_name] = L
    return L
