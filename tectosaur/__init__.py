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

from tectosaur.util.logging import setup_logger
logger = setup_logger(__name__)
