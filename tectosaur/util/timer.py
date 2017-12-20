import tectosaur.util.logging as tct_log
import time

#TODO: This has gotten WAY too complex. Replace with the simplified timer in
# taskloaf, or just import that one!
class Timer(object):
    def __init__(self, just_print = False, tabs = 0, silent = False, prefix = "", logger = None):
        self.just_print = just_print
        self.tabs = tabs
        self.silent = silent
        self.start = time.time()
        self.prefix = prefix
        self.logger = logger
        if not just_print and self.logger is None:
            self.logger = tct_log.get_caller_logger()

    def write(self, text):
        if self.just_print:
            print(text)
        else:
            self.logger.debug(text)

    def restart(self):
        self.start = time.time()

    def report(self, name, should_restart = True):
        if not self.silent:
            text = '    ' * self.tabs
            if self.prefix is not "":
                text += self.prefix + ' -- '
            text += name + " took "
            text += str(time.time() - self.start)
            self.write(text)
        if should_restart:
            self.restart()
