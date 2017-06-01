import logging
import sys
import os
source_dir = os.path.dirname(os.path.realpath(__file__))

from .mesh import concat, make_rect, refine, refine_to_size
from .constraints import continuity_constraints, constant_bc_constraints, free_edge_constraints, \
    build_constraint_matrix, all_bc_constraints
from .sparse_integral_op import SparseIntegralOp
from .combined_mesh import CombinedMesh

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
