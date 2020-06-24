import os

###
# This file adds links to source project model and data directories
###



ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..'))
ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..'))

EXPERIMENTS_ROOT_DIR = os.path.join(ROOT_DIR, 'models')


DATA_DIR = os.path.join(ROOT_DIR, 'data')
