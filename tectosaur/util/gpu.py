import numpy as np
import pyopencl as cl
import pyopencl.array
import os


gpu_initialized = False
gpu_ctx = None
gpu_queue = None
gpu_module = dict()

def initialize_with_ctx(ctx):
    global gpu_initialized, gpu_ctx, gpu_queue
    gpu_ctx = ctx
    gpu_queue = cl.CommandQueue(
        gpu_ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE
    )
    gpu_initialized = True

    # Lazy import to avoid a circular dependency
    import tectosaur.viennacl as viennacl
    viennacl.setup(gpu_ctx.int_ptr, gpu_ctx.devices[0].int_ptr, gpu_queue.int_ptr)

def check_initialized():
    global gpu_initialized
    if not gpu_initialized:
        initialize_with_ctx(cl.create_some_context())

def to_gpu(arr, float_type = np.float32):
    check_initialized()
    if type(arr) is cl.array.Array:
        return arr
    return cl.array.to_device(gpu_queue, arr.astype(float_type))

def zeros_gpu(shape, float_type = np.float32):
    check_initialized()
    return cl.array.zeros(gpu_queue, shape, float_type)

def empty_gpu(shape, float_type = np.float32):
    check_initialized()
    return cl.array.empty(gpu_queue, shape, float_type)

def quad_to_gpu(quad_rule, float_type = np.float32):
    gpu_qx = to_gpu(quad_rule[0].flatten(), float_type)
    gpu_qw = to_gpu(quad_rule[1], float_type)
    return gpu_qx, gpu_qw

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

def get_tectosaur_dir():
    import tectosaur
    tectosaur_dir = os.path.dirname(tectosaur.__file__)
    return tectosaur_dir


def load_gpu(tmpl_name, tmpl_dir = None, print_code = False,
        no_caching = False, tmpl_args = None):


    if tmpl_args is None:
        tmpl_args = dict()

    check_initialized()

    if tmpl_name in gpu_module \
            and not no_caching \
            and not print_code:

        existing_modules = gpu_module[tmpl_name]
        for module_info in existing_modules:
            tmpl_args_match = True
            for k, v in module_info['tmpl_args'].items():
                tmpl_args_match = tmpl_args_match and compare(v, tmpl_args[k])

            if tmpl_args_match:
                return module_info['module']

    import mako.exceptions
    import mako.lookup

    def get_template(tmpl_name, tmpl_dir):
        template_dirs = [get_tectosaur_dir()]
        if tmpl_dir is not None:
            template_dirs.append(tmpl_dir)
        lookup = mako.lookup.TemplateLookup(directories = template_dirs)
        return lookup.get_template(tmpl_name)


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
    debug_opts = ['-g']
    # compile_options.extend(debug_opts)
    # Using these optimization options doesn't improve performance by very much, if any,
    # so I'd say they're not worth the risk.
    # fast_opts = [
    #     '-cl-finite-math-only',
    #     '-cl-unsafe-math-optimizations',
    #     '-cl-no-signed-zeros',
    #     '-cl-mad-enable',
    #     '-cl-strict-aliasing'
    # ]
    # compile_options.extend(fast_opts)
    module_info['module'] = cl.Program(
        gpu_ctx, code
    ).build(options = compile_options)

    if tmpl_name not in gpu_module:
        gpu_module[tmpl_name] = []
    gpu_module[tmpl_name].append(module_info)

    return module_info['module']
