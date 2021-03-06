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
from tectosaur.util.timer import Timer

from tectosaur.mesh.mesh_gen import make_rect
from tectosaur.mesh.combined_mesh import CombinedMesh
from tectosaur.mesh.modify import concat
from tectosaur.mesh.refine import refine

from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.sum_op import SumOp
from tectosaur.ops.neg_op import NegOp, MultOp
from tectosaur.ops.composite_op import CompositeOp
from tectosaur.ops.sparse_integral_op import RegularizedSparseIntegralOp
from tectosaur.ops.sparse_farfield_op import (
    TriToTriDirectFarfieldOp,
    FMMFarfieldOp)
from tectosaur.ops.dense_integral_op import RegularizedDenseIntegralOp
from tectosaur.interior import InteriorOp

from tectosaur.constraints import Term, ConstraintEQ, build_constraint_matrix, \
    simple_constraint_matrix
from tectosaur.constraint_builders import (
    free_edge_constraints,
    simple_constraints,
    find_free_edges,
    free_edge_dofs,
    build_composite_constraints,
    jump_constraints,
    check_continuity,
    all_bc_constraints)
from tectosaur.continuity import continuity_constraints, traction_admissibility_constraints
