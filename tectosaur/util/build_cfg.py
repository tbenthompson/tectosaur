import numpy as np
import os

from cppimport import setup_pybind11, turn_off_strict_prototypes

import tectosaur
from tectosaur.util.gpu import np_to_c_type

def get_fmm_dir():
    return os.path.join(tectosaur.source_dir, 'fmm')

def to_fmm_dir(filenames):
    return [os.path.join(get_fmm_dir(), fname) for fname in filenames]

compiler_args = [
    '-std=c++14', '-O3', '-g', '-Wall', '-Werror', '-fopenmp', '-UNDEBUG', '-DDEBUG'
]
linker_args = ['-fopenmp']

def setup_module(cfg):
    turn_off_strict_prototypes()
    setup_pybind11(cfg)
    cfg['compiler_args'] += compiler_args
    cfg['parallel'] = False #TODO: Why is parallel building broken?
    cfg['linker_args'] += linker_args
    cfg['include_dirs'] += [tectosaur.source_dir]
    cfg['dependencies'] += [os.path.join(tectosaur.source_dir, 'util', 'build_cfg.py')]

def fmm_lib_cfg(cfg):
    setup_module(cfg)
    cfg['sources'] += to_fmm_dir(['traversal.cpp', 'octree.cpp', 'kdtree.cpp'])
    cfg['dependencies'] += to_fmm_dir([
        'traversal.hpp', 'octree.hpp', 'kdtree.hpp', 'tree_helpers.hpp',
        'traversal.cpp',
        os.path.join(tectosaur.source_dir, 'include', 'pybind11_nparray.hpp'),
    ])

def fmm_test_cfg(cfg):
    fmm_lib_cfg(cfg)
    cfg['sources'] += ['test_octree.cpp']
    cfg['dependencies'] += [
        os.path.join(tectosaur.source_dir, 'include', 'test_helpers.hpp'),
        os.path.join(tectosaur.source_dir, 'include', 'doctest.h')
    ]
    cfg['include_dirs'] += [get_fmm_dir()]
