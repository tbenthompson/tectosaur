import pycuda.autoinit
from pycuda.compiler import SourceModule
import os

import mako.template
import mako.runtime
import mako.exceptions
import mako.lookup

def load_gpu(filepath, print_code = False):
    lookup = mako.lookup.TemplateLookup(directories=[os.path.dirname(filepath)])
    tmpl = mako.template.Template(filename = filepath, lookup = lookup)
    try:
        code = tmpl.render()
    except:
        print(mako.exceptions.text_error_template().render())
    if print_code:
        numbered_lines = '\n'.join(
            [str(i) + '   ' + line for i,line in enumerate(code.split('\n'))]
        )
        print(numbered_lines)
    mod = SourceModule(
        code, options = ['-std=c++11'], no_extern_c = True,
        include_dirs = [os.getcwd() + '/' + os.path.dirname(filepath)]
    )
    return mod
