import tectosaur.util.logging as tct_log
import time

class Timer(object):
    def __init__(self, tabs = 0, silent = False, prefix = "", logger = None):
        self.tabs = tabs
        self.silent = silent
        self.start = time.time()
        self.prefix = prefix
        self.logger = logger
        if self.logger is None:
            self.logger = tct_log.get_caller_logger()

    def restart(self):
        self.start = time.time()

    def report(self, name, should_restart = True):
        if not self.silent:
            text = '    ' * self.tabs
            if self.prefix is not "":
                text += self.prefix + ' -- '
            text += name + " took "
            text += str(time.time() - self.start)
            self.logger.debug(text)
        if should_restart:
            self.restart()
