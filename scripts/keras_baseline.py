import os
import argparse
import joblib
import numpy as np
from tqdm import tqdm
import imgaug as ia

from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam, Adamax
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import multi_gpu_model

from sklearn.metrics import f1_score as off1

from tensorflow import set_random_seed

from utils.keras import f1_loss, f1
from utils.keras_models import baseline1
from utils.keras_image import ProteinDataGenerator
from utils.data import *
from utils import defines

TAG = 'v2'
BATCH_SIZE = 32
EPOCHS = 1
SEED = 42
SHAPE = (512, 512, 4)
THRESHOLD = 0.05
DEFAULT_FOLD_PATH = os.path.join(defines.SPLIT_DIR, 'stratified_10folds_seed42')

# Set the seeds for reproducibility
ia.seed(SEED)
set_random_seed(SEED)
np.random.seed(SEED)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Run baseline Keras model', add_help=True)
    parser.add_argument('--fold_path', action='store', dest='fold_path',
                        help='Path to pickle files containing the (train, test) folds',
                        type=str,
                        default=DEFAULT_FOLD_PATH)
    parser.add_argument('-n', '--fold', action='store', dest='fold',
                        help='Number of fold',
                        type=int,
                        default=0)
    parser.add_argument('-g', '--gpu', action='store', dest='gpu',
                        help='Which gpu to use: 0, 1 or 2 for both.',
                        type=int,
                        choices=[0, 1, 2],
                        default=0)
    options = parser.parse_args()

    # Set the device to be used. This has to be done before any call to Keras
    if options.gpu < 2:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu)
        # from keras import backend as K
        # K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8,
        #                                                    inter_op_parallelism_threads=8)))

    # Create the model
    model = baseline1(input_shape=SHAPE)
    if options.gpu == 2:
        model = multi_gpu_model(model, gpus=2, cpu_relocation=True, cpu_merge=True)

    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # opt = Adam(lr=1.E-02)
    opt = Adamax(decay=1.E-06)

    # Compile
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['acc', f1])
    model.summary()

    # Get train dataset and labels
    paths, labels = get_train_dataset()

    # Split train/validation
    (train_idxs, val_idxs) = joblib.load(os.path.join(options.fold_path, 'fold_%s.pkl' % str(options.fold)))
    paths_train = paths[train_idxs]
    labels_train = labels[train_idxs]
    paths_val = paths[val_idxs]
    labels_val = labels[val_idxs]

    print(paths.shape, labels.shape)
    print(paths_train.shape, labels_train.shape, paths_val.shape, labels_val.shape)

    tg = ProteinDataGenerator(paths_train, labels_train, BATCH_SIZE, SHAPE, use_cache=False, augment=False, shuffle=False)
    vg = ProteinDataGenerator(paths_val, labels_val, BATCH_SIZE, SHAPE, use_cache=False, augment=False, shuffle=False)

    # https://keras.io/callbacks/#modelcheckpoint
    checkpoint = ModelCheckpoint('./base.model', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.001, factor=0.5, patience=5, verbose=2, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min')

    import time
    t0 = time.time()
    hist = model.fit_generator(
        tg,
        steps_per_epoch=len(tg),
        validation_data=vg,
        validation_steps=len(vg),
        epochs=EPOCHS,
        use_multiprocessing=False,
        workers=1,
        verbose=0,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    t1 = time.time()
    print('Training took {0:.2f} seconds.'.format((t1-t0)))
    import sys
    sys.exit(0)

    bestModel = model

    fullValGen = vg

    lastFullValPred = np.empty((0, 28))
    lastFullValLabels = np.empty((0, 28))
    for i in tqdm(range(len(fullValGen))):
        im, lbl = fullValGen[i]
        scores = bestModel.predict(im)
        lastFullValPred = np.append(lastFullValPred, scores, axis=0)
        lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)
    print(lastFullValPred.shape, lastFullValLabels.shape)

    rng = np.arange(0, 1, 0.001)
    f1s = np.zeros((rng.shape[0], 28))
    for j, t in enumerate(tqdm(rng)):
        for i in range(28):
            p = np.array(lastFullValPred[:, i] > t, dtype=np.int8)
            scoref1 = off1(lastFullValLabels[:, i], p, average='binary')
            f1s[j, i] = scoref1

    print('Individual F1-scores for each class:')
    print(np.max(f1s, axis=0))
    print('Macro F1-score CV =', np.mean(np.max(f1s, axis=0)))

    T = np.empty(28)
    for i in range(28):
        T[i] = rng[np.where(f1s[:, i] == np.max(f1s[:, i]))[0][0]]
    print('Probability threshold maximizing CV F1-score for each class:')
    print(T)

    pathsTest, labelsTest = get_test_dataset()

    testg = ProteinDataGenerator(pathsTest, labelsTest, BATCH_SIZE, SHAPE)
    submit = pd.read_csv(defines.DATA_DIR + '/sample_submission.csv')
    P = np.zeros((pathsTest.shape[0], 28))
    for i in tqdm(range(len(testg))):
        images, labels = testg[i]
        score = bestModel.predict(images)
        P[i * BATCH_SIZE:i * BATCH_SIZE + score.shape[0]] = score

    PP = np.array(P)

    prediction = []

    for row in tqdm(range(submit.shape[0])):

        str_label = ''

        for col in range(PP.shape[1]):
            if (PP[row, col] < T[col]):
                str_label += ''
            else:
                str_label += str(col) + ' '
        prediction.append(str_label.strip())

    submit['Predicted'] = np.array(prediction)
    submit.to_csv('../submissions/4channels_cnn_from_scratch_epochs%s_%s.csv' % (str(EPOCHS), TAG), index=False)
