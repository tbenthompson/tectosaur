import os
from joblib import Memory

cachedir = os.path.join(os.getcwd(), 'cache_tectosaur')
memory = Memory(cachedir = cachedir, verbose = 0)
cache = memory.cache
