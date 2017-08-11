import os
import pyopencl
import warnings
import numpy as np

from tectosaur.util.logging import setup_logger
logger = setup_logger(__name__)

gpu_initialized = False
gpu_ctx = None
gpu_queue = None

def report_devices(ctx):
    device_names = [d.name for d in ctx.devices]
    logger.info('initializing opencl context with devices = ' + str(device_names))

def initialize_with_ctx(ctx):
    global gpu_initialized, gpu_ctx, gpu_queue
    gpu_ctx = ctx
    gpu_queue = pyopencl.CommandQueue(
        gpu_ctx,
        properties=pyopencl.command_queue_properties.PROFILING_ENABLE
    )
    gpu_initialized = True

    # Lazy import to avoid a circular dependency
    import tectosaur.viennacl.viennacl as vcl
    vcl.setup(gpu_ctx.int_ptr, gpu_ctx.devices[0].int_ptr, gpu_queue.int_ptr)

    report_devices(ctx)

def ensure_initialized():
    global gpu_initialized
    if not gpu_initialized:
        initialize_with_ctx(pyopencl.create_some_context())

def to_gpu(arr, float_type = np.float32):
    ensure_initialized()
    if type(arr) is pyopencl.array.Array:
        return arr
    to_type = arr.astype(float_type)
    # tct_log.get_caller_logger().debug('sending n_bytes: ' + str(to_type.nbytes))
    return pyopencl.array.to_device(gpu_queue, to_type)

def zeros_gpu(shape, float_type = np.float32):
    ensure_initialized()
    return pyopencl.array.zeros(gpu_queue, shape, float_type)

def empty_gpu(shape, float_type = np.float32):
    ensure_initialized()
    return pyopencl.array.empty(gpu_queue, shape, float_type)

class ModuleWrapper:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        kernel = getattr(self.module, name)
        def provide_queue_wrapper(*args, **kwargs):
            return kernel(gpu_queue, *args, **kwargs)
        return provide_queue_wrapper

def opencl_compile(code):
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
        return ModuleWrapper(pyopencl.Program(
            gpu_ctx, code
        ).build(options = compile_options))
