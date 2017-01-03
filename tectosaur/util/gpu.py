import numpy as np
import pyopencl as cl
import pyopencl.array
import os

import mako.template
import mako.runtime
import mako.exceptions
import mako.lookup

from tectosaur.util.timer import Timer

def compare(a, b):
    if type(a) != type(b):
        return False
    if type(a) is list or type(a) is tuple:
        if len(a) != len(b):
            return False
        comparisons = [compare(av, bv) for av, bv in zip(a,b)]
        if False in comparisons:
            return False
        return True
    if type(a) is np.ndarray:
        return (a == b).all()
    return a == b

ocl_gpu_initialized = False
ocl_gpu_ctx = None
ocl_gpu_queue = None
ocl_gpu_module = dict()

def to_gpu(arr, float_type = np.float32):
    if type(arr) is cl.array.Array:
        return arr
    return cl.array.to_device(ocl_gpu_queue, arr.astype(float_type))

def empty_gpu(shape, float_type = np.float32):
    return cl.array.zeros(ocl_gpu_queue, shape, float_type)

def quad_to_gpu(quad_rule, float_type = np.float32):
    gpu_qx = to_gpu(quad_rule[0].flatten(), float_type)
    gpu_qw = to_gpu(quad_rule[1], float_type)
    return gpu_qx, gpu_qw

def intervals(length, step_size):
    out = []
    next_start = 0
    next_end = step_size
    while next_end < length + step_size:
        this_end = min(next_end, length)
        out.append((next_start, this_end))
        next_start += step_size
        next_end += step_size
    return out


def ocl_load_gpu(filepath, print_code = False, no_caching = False, tmpl_args = None):
    global ocl_gpu_initialized, ocl_gpu_ctx, ocl_gpu_queue, ocl_gpu_module

    if tmpl_args is None:
        tmpl_args = dict()

    if not ocl_gpu_initialized:
        ocl_gpu_ctx = cl.create_some_context()
        ocl_gpu_queue = cl.CommandQueue(ocl_gpu_ctx)
        ocl_gpu_initialized = True

    if filepath in ocl_gpu_module \
            and not no_caching \
            and not print_code:

        existing_modules = ocl_gpu_module[filepath]
        for module_info in existing_modules:
            tmpl_args_match = True
            for k, v in module_info['tmpl_args'].items():
                tmpl_args_match = tmpl_args_match and compare(v, tmpl_args[k])

            if tmpl_args_match:
                return module_info['module']

    lookup = mako.lookup.TemplateLookup(
        directories = [os.path.dirname(filepath)]
    )
    tmpl = mako.template.Template(
        filename = filepath,
        module_directory = '/tmp/mako_modules',
        lookup = lookup
    )

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

    module_info = dict()
    module_info['tmpl_args'] = tmpl_args
    #TODO: set options for efficiency -- https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clBuildProgram.html
    file_dir = os.getcwd() + '/' + os.path.dirname(filepath)
    compile_options = ['-I' + file_dir]
    module_info['module'] = cl.Program(
        ocl_gpu_ctx, code
    ).build(options = compile_options)

    if filepath not in ocl_gpu_module:
        ocl_gpu_module[filepath] = []
    ocl_gpu_module[filepath].append(module_info)

    return module_info['module']
