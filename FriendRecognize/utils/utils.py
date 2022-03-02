import copy
import pickle as pk
from enum import Enum

import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from FriendRecognize.utils.crop_and_filter import preprocessing
from FriendRecognize.utils.hog.hog_feature import HOG
from FriendRecognize.utils.lbp.lbp_feature import LBP


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
def extraction_feature_HOG(X, y, kind_of_feature):
    hog = HOG(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    x_new = []
    for i in tqdm(range(X.shape[0]), desc="Extract feature for " + str(kind_of_feature)):
        hist, _ = hog.describe(X[i])
        x_new.append(hist)
    x_new = np.array(x_new)
    return x_new, y


# LBP
def extraction_feature_LBP(X, y, kind_of_feature):
    lbp = LBP(numPoints=16, radius=2, num_bins=100, n_row=8, n_col=8)
    x_new = []
    for i in tqdm(range(X.shape[0]), desc=kind_of_feature):
        hist, _ = lbp.describe(X[i])
        x_new.append(hist)
    x_new = np.array(x_new)
    return x_new, y


def extraction_feature_PCA(X, y, kind_of_feature):
    # #############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction

    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples, nx * ny))

    n_components = 40

    pca = PCA(n_components=n_components).fit(X)

    x_new = pca.transform(X)

    return x_new, y


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
    # X_train, y_train = extraction_feature_LBP(X_train,
    #                                          y_train,
    #                                          "Extract feature: trainingSet-set of " + str(kind_of_feature))
    # X_val, y_val = extraction_feature_LBP(X_val,
    #                                      y_val,
    #                                      "Extract feature: validation-set of " + str(kind_of_feature))

    X_train, y_train = extraction_feature_LBP(X_train,
                                              y_train,
                                              "Extract feature: trainingSet-set of " + str(kind_of_feature))
    X_val, y_val = extraction_feature_LBP(X_val,
                                          y_val,
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


# --> FUNCTIONS FOR TESTING <--

def compute_feature_for_testing(X_val, y_val, kind_of_feature):
    print("\n ---> Extract feature validation set <--- ")
    if kind_of_feature is not Feature.GLASSES:
        X_val, y_val = extraction_feature_LBP(X_val,
                                              y_val,
                                              "Extract feature: validation-set of " + str(kind_of_feature))
    else:
        X_val, y_val = extraction_feature_HOG(X_val,
                                              y_val,
                                              "Extract feature: validation-set of " + str(kind_of_feature))
    return X_val, y_val


def extract_features_for_testing(img, label_value, detector, predictor, kind_of_feature):
    prep_img = preprocessing(img, detector, predictor, kind_of_feature)
    X = []
    y = []
    X.append(prep_img)
    y.append(label_value)
    return compute_feature_for_testing(np.array(X), np.array(y), kind_of_feature)


def get_labels_from(csv, csv_path):
    results = {}
    with open(csv_path, mode='r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        for row in rows:
            results.update({row[0]: {
                Feature.BEARD.value: int(round(float(row[1]))),
                Feature.MUSTACHE.value: int(round(float(row[2]))),
                Feature.GLASSES.value: int(round(float(row[3])))
            }
            })
    return results
