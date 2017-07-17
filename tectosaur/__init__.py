import logging
import sys
import os
import numpy as np

source_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(source_dir, 'data')

float_type = np.float32
if float_type == np.float64:
    print('warning: float type is set to double precision')

def get_data_filepath(filename):
    return os.path.join(data_dir, filename)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

from .mesh.mesh_gen import make_rect, make_sphere
from .mesh.modify import concat
from .mesh.refine import refine, refine_to_size, selective_refine

from .ops.sparse_integral_op import SparseIntegralOp
# from .ops.fmm_integral_op import FMMIntegralOp
from .ops.mass_op import MassOp
from .ops.neg_op import NegOp
from .ops.sum_op import SumOp

from .constraints import continuity_constraints, constant_bc_constraints, free_edge_constraints, \
    build_constraint_matrix, all_bc_constraints
