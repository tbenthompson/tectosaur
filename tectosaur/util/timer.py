import time

class Timer(object):
    def __init__(self, tabs = 0, silent = False):
        self.tabs = tabs
        self.silent = silent
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def report(self, name, should_restart = True):
        if not self.silent:
            print('    ' * self.tabs + name + " took " + str(time.time() - self.start))
        if should_restart:
            self.restart()
