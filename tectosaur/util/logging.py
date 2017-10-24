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
    formatter = logging.Formatter(
        # "[%(asctime)s:%(levelname)8s:%(name)35s] %(message)s",
        # "[%(asctime)s:%(levelname)s:%(name)s]\n    %(message)s",
        "[%(relativeCreated)d:%(levelname)s:%(name)s]\n    %(message)s",
        datefmt='%j:%H:%M:%S'
    )
    ch.setFormatter(formatter)
    L.addHandler(ch)
    return L
