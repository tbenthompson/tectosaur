import pycuda.autoinit
from pycuda.compiler import SourceModule
import os

import mako.template
import mako.runtime
import mako.exceptions
import mako.lookup

from tectosaur.util.timer import Timer

gpu_module = dict()
def load_gpu(filepath, print_code = False, no_caching = False, tmpl_args = None):
    global gpu_module
    if filepath in gpu_module \
            and not no_caching \
            and not print_code \
            and gpu_module[filepath]['tmpl_args'] == tmpl_args:
        return gpu_module[filepath]['module']

    timer = Timer(silent = True)
    if tmpl_args is None:
        tmpl_args = dict()

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
        code = tmpl.render(**tmpl_args)
        # print(code)
    except:
        print(mako.exceptions.text_error_template().render())

    if print_code:
        numbered_lines = '\n'.join(
            [str(i) + '   ' + line for i,line in enumerate(code.split('\n'))]
        )
        print(numbered_lines)
    timer.report("Render template")
    gpu_module[filepath] = dict()
    gpu_module[filepath]['tmpl_args'] = tmpl_args
    gpu_module[filepath]['module'] = SourceModule(
        code,
        options = ['--use_fast_math', '--restrict'],
        include_dirs = [os.getcwd() + '/' + os.path.dirname(filepath)],
        no_extern_c = True #TODO: This will break some other modules!
    )
    timer.report("Compiling cuda")
    return gpu_module[filepath]['module']
