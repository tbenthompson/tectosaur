import pycuda.autoinit
from pycuda.compiler import SourceModule
import os

import mako.template
import mako.runtime
import mako.exceptions
import mako.lookup

from tectosaur.util.timer import Timer

gpu_module = dict()
def load_gpu(filepath, print_code = False, no_caching = False):
    timer = Timer(silent = True)
    global gpu_module
    if filepath in gpu_module and no_caching and not print_code:
        return gpu_module[filepath]
    lookup = mako.lookup.TemplateLookup(
        directories = [os.path.dirname(filepath)]
    )
    tmpl = mako.template.Template(
        filename = filepath,
        module_directory = '/tmp/mako_modules',
        lookup = lookup
    )
    timer.report("Load template")

    try:
        code = tmpl.render()
    except:
        print(mako.exceptions.text_error_template().render())

    if print_code:
        numbered_lines = '\n'.join(
            [str(i) + '   ' + line for i,line in enumerate(code.split('\n'))]
        )
        print(numbered_lines)
    timer.report("Render template")
    gpu_module[filepath] = SourceModule(
        code, options = [],
        include_dirs = [os.getcwd() + '/' + os.path.dirname(filepath)]
    )
    timer.report("Compiling cuda")
    return gpu_module[filepath]
