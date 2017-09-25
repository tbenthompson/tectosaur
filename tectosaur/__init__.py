import sys
import os
import logging
import numpy as np

source_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(source_dir, 'data')

def get_data_filepath(filename):
    return os.path.join(data_dir, filename)

from tectosaur.util.logging import setup_root_logger
logger = setup_root_logger(__name__)
