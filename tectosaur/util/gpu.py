import os
import warnings

import numpy as np
import pyopencl as cl
import pyopencl.array

import tectosaur
import tectosaur.util.logging as tct_log

logger = tct_log.setup_logger(__name__)

gpu_initialized = False
gpu_ctx = None
gpu_queue = None
gpu_module = dict()

class ContextBuilder:
    get_platforms = lambda self: pyopencl.get_platforms()

    def from_platform(self, platform, device_idx):
        devices = platform.get_devices()
        assert(len(devices) > 0)
        return pyopencl.Context(devices = [devices[device_idx]])

    def check_for_env_variable(self):
        return os.environ.get('PYOPENCL_CTX', '') != ''

    def make_any_context(self):
        return pyopencl.create_some_context()

def make_default_ctx(ctx_builder = ContextBuilder(), device_idx = 0):
    if ctx_builder.check_for_env_variable():
        return ctx_builder.make_any_context()

    platforms = ctx_builder.get_platforms()
    gpu_platforms = [p for p in platforms if 'CUDA' in p.name]
    if len(gpu_platforms) > 0:
        assert(len(gpu_platforms) == 1)
        return ctx_builder.from_platform(gpu_platforms[0], device_idx)

    other_platforms = [p for p in platforms if 'CUDA' not in p.name]
    if len(other_platforms) > 0:
        return ctx_builder.from_platform(other_platforms[0], device_idx)

    raise Exception("No OpenCL platforms available")

def report_devices(ctx):
    device_names = [d.name for d in ctx.devices]
    logger.info('initializing opencl context with devices = ' + str(device_names))

def initialize_with_ctx(ctx):
    global gpu_initialized, gpu_ctx, gpu_queue
    gpu_ctx = ctx
    gpu_queue = cl.CommandQueue(
        gpu_ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE
    )
    gpu_initialized = True

    # Lazy import to avoid a circular dependency
    import tectosaur.viennacl.viennacl as vcl
    vcl.setup(gpu_ctx.int_ptr, gpu_ctx.devices[0].int_ptr, gpu_queue.int_ptr)

    report_devices(ctx)

def ensure_initialized():
    global gpu_initialized
    if not gpu_initialized:
        initialize_with_ctx(make_default_ctx())

def to_gpu(arr, float_type = np.float32):
    ensure_initialized()
    if type(arr) is cl.array.Array:
        return arr
    to_type = arr.astype(float_type)
    # tct_log.get_caller_logger().debug('sending n_bytes: ' + str(to_type.nbytes))
    return cl.array.to_device(gpu_queue, to_type)

def zeros_gpu(shape, float_type = np.float32):
    ensure_initialized()
    return cl.array.zeros(gpu_queue, shape, float_type)

def empty_gpu(shape, float_type = np.float32):
    ensure_initialized()
    return cl.array.empty(gpu_queue, shape, float_type)

def np_to_c_type(t):
    if t == np.float32:
        return 'float'
    elif t == np.float64:
        return 'double'

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

def load_gpu(tmpl_name, tmpl_dir = None, print_code = False,
        no_caching = False, tmpl_args = None):

    logger = tct_log.get_caller_logger()

    if tmpl_args is None:
        tmpl_args = dict()

    ensure_initialized()

    if tmpl_name in gpu_module \
            and not no_caching \
            and not print_code:

        existing_modules = gpu_module[tmpl_name]
        for module_info in existing_modules:
            tmpl_args_match = True
            for k, v in module_info['tmpl_args'].items():
                tmpl_args_match = tmpl_args_match and compare(v, tmpl_args[k])

            if tmpl_args_match:
                logger.debug('returning cached gpu module ' + tmpl_name)
                return module_info['module']

    import mako.exceptions
    import mako.lookup

    def get_template(tmpl_name, tmpl_dir):
        template_dirs = [tectosaur.source_dir]
        if tmpl_dir is not None:
            template_dirs.append(tmpl_dir)
        lookup = mako.lookup.TemplateLookup(directories = template_dirs)
        return lookup.get_template(tmpl_name)


    import time
    start = time.time()
    logger.debug('start compiling ' + tmpl_name)

    tmpl = get_template(tmpl_name, tmpl_dir)
    try:
        code = tmpl.render(**tmpl_args)
    except:
        print(mako.exceptions.text_error_template().render())
        raise

    if print_code:
        numbered_lines = '\n'.join(
            [str(i) + '   ' + line for i,line in enumerate(code.split('\n'))]
        )
        print(numbered_lines)

    module_info = dict()
    module_info['tmpl_args'] = tmpl_args
    compile_options = []

    debug_opts = ['-g', '-Werror']
    # compile_options.extend(debug_opts)
    fast_opts = [
        # '-cl-finite-math-only',
        '-cl-unsafe-math-optimizations',
        # '-cl-no-signed-zeros',
        '-cl-mad-enable',
        # '-cl-strict-aliasing'
    ]
    compile_options.extend(fast_opts)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pyopencl.CompilerWarning)
        module_info['module'] = cl.Program(
            gpu_ctx, code
        ).build(options = compile_options)

    logger.debug('compile took: ' + str(time.time() - start))

    if tmpl_name not in gpu_module:
        gpu_module[tmpl_name] = []
    gpu_module[tmpl_name].append(module_info)

    return module_info['module']
