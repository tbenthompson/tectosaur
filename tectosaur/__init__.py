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

def setup_logger(log_name):
    L = logging.getLogger(log_name)
    L.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    L.addHandler(ch)
    return L

logger = setup_logger(__name__)
