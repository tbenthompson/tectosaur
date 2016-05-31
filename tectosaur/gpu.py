import pycuda.autoinit
from pycuda.compiler import SourceModule
import os

import mako.template
import mako.runtime
import mako.exceptions
import mako.lookup

gpu_module = dict()
def load_gpu(filepath, print_code = False, no_caching = False):
    global gpu_module
    if filepath in gpu_module and no_caching and not print_code:
        return gpu_module[filepath]
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
    gpu_module[filepath] = SourceModule(
        code, options = [],
        include_dirs = [os.getcwd() + '/' + os.path.dirname(filepath)]
    )
    return gpu_module[filepath]
