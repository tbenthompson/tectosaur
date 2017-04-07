import time

class Timer(object):
    def __init__(self, tabs = 0, silent = False, prefix = ""):
        self.tabs = tabs
        self.silent = silent
        self.start = time.time()
        self.prefix = prefix

    def restart(self):
        self.start = time.time()

    def report(self, name, should_restart = True):
        if not self.silent:
            text = '    ' * self.tabs
            if self.prefix is not "":
                text += self.prefix + ' -- '
            text += name + " took "
            text += str(time.time() - self.start)
            print(text)
        if should_restart:
            self.restart()
