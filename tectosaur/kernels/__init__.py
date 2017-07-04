class Kernel:
    def __init__(self, name, needs_obsn, needs_srcn):
        self.name = name
        self.needs_obsn = needs_obsn
        self.needs_srcn = needs_srcn

kernels = [
    Kernel('elasticU', False, False),
    Kernel('elasticT', False, True),
    Kernel('elasticA', True, False),
    Kernel('elasticH', True, True)
]
