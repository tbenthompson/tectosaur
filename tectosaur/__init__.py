import cppimport
import ctypes
cppimport.set_rtld_flags(ctypes.RTLD_GLOBAL)

import os

source_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(source_dir, 'data')

def get_data_filepath(filename):
    return os.path.join(data_dir, filename)

from tectosaur.util.logging import setup_root_logger
logger = setup_root_logger(__name__)

from tectosaur.mesh.mesh_gen import make_rect

from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.sum_op import SumOp
from tectosaur.ops.neg_op import NegOp, MultOp
from tectosaur.ops.composite_op import CompositeOp
from tectosaur.ops.sparse_integral_op import RegularizedSparseIntegralOp
from tectosaur.ops.sparse_farfield_op import (
    TriToTriDirectFarfieldOp,
    FMMFarfieldOp)
from tectosaur.ops.dense_integral_op import RegularizedDenseIntegralOp

from tectosaur.constraints import Term, ConstraintEQ, build_constraint_matrix
from tectosaur.constraint_builders import (
    continuity_constraints,
    free_edge_constraints,
    build_composite_constraints,
    jump_constraints,
    check_continuity)
