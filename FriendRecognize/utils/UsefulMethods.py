import copy
import pickle as pk
from enum import Enum

import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from FriendRecognize.utils.CropAndFilter import preprocessing
from FriendRecognize.utils.hog.HOGFeature import HOG
from FriendRecognize.utils.lbp.LBPFeature import LBP


class Feature(Enum):
    VINCENZO = "Vincenzo"
    ANGELO = "Angelo"
    DIMA = "Dima"
    GIOVANNA = "Giovanna"
    NOEMI = "Noemi"


class ImageType(Enum):
    FEATURE = 1
    NO_FEATURE = 0


# HOG
def extraction_feature_HOG(X, kind_of_feature):
    hog = HOG(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    x_new = []
    for i in tqdm(range(X.shape[0]), desc="Extract feature for " + str(kind_of_feature)):
        hist, _ = hog.describe(X[i])
        x_new.append(hist)
    x_new = np.array(x_new)
    return x_new


# LBP
def extraction_feature_LBP(X, kind_of_feature):
    lbp = LBP(numPoints=16, radius=4, num_bins=100, n_row=10, n_col=10)
    x_new = []
    for i in tqdm(range(X.shape[0]), desc=kind_of_feature):
        hist, _ = lbp.describe(X[i])
        x_new.append(hist)
    x_new = np.array(x_new)
    return x_new


def extraction_feature_PCA(X, kind_of_feature):
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples, nx * ny))
    n_components = 40
    pca = PCA(n_components=n_components).fit(X)
    x_new = pca.transform(X)
    return x_new


# --> FUNCTIONS FOR TRAINING <--
class Labeler:
    def __init__(self, dataset=None):
        self.encoding = None
        if dataset:
            self.encoding = {}
            for id in dataset['train']:
                if id not in self.encoding:
                    self.encoding[id] = len(self.encoding)

    def encode(self, y):
        if self.encoding:
            return self.encoding[y]
        else:
            raise Exception("Encoder not initialized. Pass a data to the constructor or load a model!")

    def save(self, path_dataset, output_path='/labels.pkl'):
        with open(path_dataset + output_path, 'wb') as file:
            pk.dump(self.encoding, file)

    def load(self, path_dataset, input_path='/labels.pkl'):
        with open(path_dataset + input_path, 'rb') as file:
            self.encoding = pk.load(file)


def compute_raw_feature_for_training(X_train, y_train, X_val, y_val, kind_of_feature):
    X_train = extraction_feature_LBP(X_train,
                                     "Extract feature: trainingSet-set of " + str(kind_of_feature))
    X_val = extraction_feature_LBP(X_val,
                                   "Extract feature: validation-set of " + str(kind_of_feature))
    return X_train, y_train, X_val, y_val


def load_raw_feature_for_training(set_files, labeler, detector, predictor, kind_of_feature):
    X = []
    y = []
    images = []
    for id in tqdm(set_files, desc="Loaded identities (" + str(kind_of_feature) + ")"):
        for file in set_files[id]:
            images.append(cv.imread(file.path))
            y.append(labeler.encode(id))
    for img in tqdm(images, desc="Pre-processing " + str(kind_of_feature)):
        image_to_filter = copy.deepcopy(img)
        image = preprocessing(image_to_filter, detector, predictor)
        if image is not None:
            X.append(image)
        else:
            del y[y[len(X)]]
    return np.array(X), np.array(y)


def extract_features_for_training(dataset, labeler, detector, predictor, kind_of_feature):
    X = {}
    y = {}
    for dataset_type in ['train', 'val']:
        print(f"\n ---> Pre-processing {dataset_type} set <--- ")
        X[dataset_type], y[dataset_type] = load_raw_feature_for_training(dataset[dataset_type],
                                                                         labeler,
                                                                         detector,
                                                                         predictor,
                                                                         kind_of_feature)
    print("\n ---> Extract feature <--- ")
    return compute_raw_feature_for_training(X['train'], y['train'], X['val'], y['val'], kind_of_feature)
