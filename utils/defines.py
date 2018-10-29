import os

# Directories
PROJ_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DATA_DIR = os.path.join(PROJ_DIR, 'input')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
SUB_DIR = os.path.join(PROJ_DIR, 'submissions')

# Files
LABEL_FILE = os.path.join(DATA_DIR, 'train.csv')
SAMPLE_SUB_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')

