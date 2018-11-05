import os

# Directories
PROJ_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DATA_DIR = os.path.join(PROJ_DIR, 'input')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
SUB_DIR = os.path.join(PROJ_DIR, 'submissions')
SPLIT_DIR = os.path.join(DATA_DIR, 'splits')

# Files
LABEL_FILE = os.path.join(DATA_DIR, 'train.csv')
SAMPLE_SUB_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')

# Miscellanea
N_CLASSES = 28
CLASSES = ['Nucleoplasm', 'Nuclear membrane', 'Nucleoli', 'Nucleoli fibrillar center',
           'Nuclear speckles', 'Nuclear bodies', 'Endoplasmic reticulum', 'Golgi apparatus',
           'Peroxisomes', 'Endosomes', 'Lysosomes', 'Intermediate filaments', 'Actin filaments',
           'Focal adhesion sites', 'Microtubules', 'Microtubule ends', 'Cytokinetic bridge',
           'Mitotic spindle', 'Microtubule organizing center', 'Centrosome', 'Lipid droplets',
           'Plasma membrane', 'Cell junctions', 'Mitochondria', 'Aggresome', 'Cytosol',
           'Cytoplasmic bodies', 'Rods & rings']
CLAS_DICT = dict(zip(range(N_CLASSES), CLASSES))
