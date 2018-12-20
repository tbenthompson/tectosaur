import os

import numpy as np

import tectosaur
from tectosaur.util.timer import Timer
try:
    from tectosaur.util.cuda import compile, empty_gpu, zeros_gpu, to_gpu,\
            cluda_preamble, threaded_get
    cuda_backend = True
    ocl_backend = False
except ImportError:
    from tectosaur.util.opencl import compile, empty_gpu, zeros_gpu, to_gpu,\
            cluda_preamble, threaded_get
    cuda_backend = False
    ocl_backend = True

import logging
logger = logging.getLogger(__name__)

gpu_module = dict()

def np_to_c_type(t):
    if t == np.float32:
        return 'float'
    elif t == np.float64:
        return 'double'

# async def get(tsk_w, arr):
#     result = await tsk_w.run_in_thread(lambda: threaded_get(arr))
#     return result

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
        res = a == b
        if type(res) is np.ndarray:
            return res.all()
        else:
            return res
    return a == b

def get_existing_module(tmpl_name, tmpl_args):
    if tmpl_name not in gpu_module:
        return None

    existing_modules = gpu_module[tmpl_name]
    for module_info in existing_modules:
        tmpl_args_match = True
        for k, v in module_info['tmpl_args'].items():
            if not compare(v, tmpl_args[k]):
                tmpl_args_match = False
                break
        if tmpl_args_match:
            return module_info['module']

    return None

def get_template(tmpl_name, tmpl_dir):
    import mako.lookup
    template_dirs = [tectosaur.source_dir]
    if tmpl_dir is not None:
        template_dirs.append(tmpl_dir)
    lookup = mako.lookup.TemplateLookup(directories = template_dirs)
    return lookup.get_template(tmpl_name)


def template_with_mako(tmpl, tmpl_args):
    try:
        return tmpl.render(
            **tmpl_args, cluda_preamble = cluda_preamble,
            cuda_backend = cuda_backend, ocl_backend = ocl_backend
        )
    except:
        import mako.exceptions
        logger.error(mako.exceptions.text_error_template().render())
        raise

def save_code_to_tmp(code):
    import tempfile
    with tempfile.NamedTemporaryFile(mode = 'w', delete = False) as temp:
        temp.write(code)
        logger.info('gpu module code written to ' + temp.name)

def load_gpu(tmpl_name, tmpl_dir = None, save_code = False,
        no_caching = False, tmpl_args = None):
    if tmpl_args is None:
        tmpl_args = dict()

    if not no_caching and not save_code:
        existing_module = get_existing_module(tmpl_name, tmpl_args)
        if existing_module is not None:
            logger.debug('returning cached gpu module ' + tmpl_name)
            return existing_module

    tmpl = get_template(tmpl_name, tmpl_dir)
    return compile_module(tmpl, tmpl_name, save_code, tmpl_args)

def load_gpu_from_code(code, save_code = False, tmpl_args = None):
    from mako.template import Template
    tmpl = Template(code)
    return compile_module(tmpl, 'anonymous', save_code, tmpl_args)

def compile_module(tmpl, tmpl_name, save_code, tmpl_args):
    if tmpl_args is None:
        tmpl_args = dict()

    code = template_with_mako(tmpl, tmpl_args)

    t = Timer(output_fnc = logger.debug)
    logger.debug('start compiling ' + tmpl_name)

    if save_code:
        save_code_to_tmp(code)

    module_info = dict()
    module_info['tmpl_args'] = tmpl_args
    module_info['module'] = compile(code)
    t.report('compile')

    gpu_module[tmpl_name] = gpu_module.get(tmpl_name, []) + [module_info]
    return module_info['module']
