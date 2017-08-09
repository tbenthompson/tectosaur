import numpy as np
import os

from cppimport import setup_pybind11, turn_off_strict_prototypes

import tectosaur
import tectosaur.fmm
from tectosaur.util.gpu import np_to_c_type

#TODO: REMOVE!
float_type = np.float32
if float_type == np.float64:
    print('warning: float type is set to double precision')

gpu_float_type = np_to_c_type(float_type)

def to_fmm_dir(filenames):
    return [os.path.join(tectosaur.fmm.source_dir, fname) for fname in filenames]

compiler_args = [
    '-std=c++14', '-O3', '-g', '-Wall', '-Werror', '-fopenmp', '-UNDEBUG', '-DDEBUG'
]
linker_args = ['-fopenmp']

def setup_module(cfg):
    turn_off_strict_prototypes()
    setup_pybind11(cfg)
    cfg['compiler_args'] += compiler_args
    cfg['parallel'] = True
    cfg['linker_args'] += linker_args
    cfg['include_dirs'] += [tectosaur.source_dir]
    cfg['dependencies'] += [os.path.join(tectosaur.source_dir, 'util', 'build_cfg.py')]

def fmm_lib_cfg(cfg):
    setup_module(cfg)
    cfg['sources'] += to_fmm_dir(['fmm_impl.cpp', 'octree.cpp', 'kdtree.cpp'])
    cfg['dependencies'] += to_fmm_dir([
        'fmm_impl.hpp', 'octree.hpp', 'kdtree.hpp',
        os.path.join(tectosaur.source_dir, 'include', 'pybind11_nparray.hpp'),
    ])

def fmm_test_cfg(cfg):
    fmm_lib_cfg(cfg)
    cfg['sources'] += ['test_octree.cpp']
    cfg['dependencies'] += ['test_helpers.hpp', 'doctest.h']
    cfg['include_dirs'] += [tectosaur.fmm.source_dir]
