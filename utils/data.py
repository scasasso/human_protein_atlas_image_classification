import os
import numpy as np
import pandas as pd
from utils import defines


def get_train_dataset():
    data = pd.read_csv(defines.LABEL_FILE)

    paths = []
    labels = []

    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(defines.N_CLASSES)
        for key in lbl:
            y[int(key)] = 1
        paths.append(os.path.join(defines.TRAIN_DIR, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


def get_test_dataset():
    data = pd.read_csv(defines.SAMPLE_SUB_FILE)

    paths = []
    labels = []

    for name in data['Id']:
        y = np.ones(defines.N_CLASSES)
        paths.append(os.path.join(defines.TEST_DIR, name))
        labels.append(y)

    return np.array(paths), np.array(labels)
