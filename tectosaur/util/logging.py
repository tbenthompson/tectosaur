import logging
import sys

def get_caller_logger():
    import inspect
    module = None
    start_idx = 2
    stack = inspect.stack()
    while module is None and start_idx < len(stack):
        frm = stack[start_idx]
        module = inspect.getmodule(frm[0])
        start_idx += 1
    mod_name = module.__name__
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
