from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils.data import *
from utils import defines
import joblib

SEED = 42
N_FOLD = 10
TAG = 'stratififed_v1'
SPLIT_DIR = os.path.join(defines.DATA_DIR, 'splits')

# Get train dataset and labels
paths, labels = get_train_dataset()

# Create directory
if not os.path.exists(SPLIT_DIR):
    os.makedirs(SPLIT_DIR)

# K-fold object
mskf = MultilabelStratifiedKFold(n_splits=N_FOLD, random_state=SEED, shuffle=True)

# Loop
for ifold, (train_index, test_index) in enumerate(mskf.split(paths, labels)):
    out_name = '_'.join([TAG, 'seed' + str(SEED), str(ifold), str(N_FOLD)])
    out_name += '.pkl'
    out_name = os.path.join(SPLIT_DIR, out_name)
    joblib.dump((list(train_index), list(test_index)), out_name)
