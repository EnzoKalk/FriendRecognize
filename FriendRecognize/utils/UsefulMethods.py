import copy
import os
import pickle as pk
from enum import Enum

import cv2 as cv
import numpy as np
from tqdm import tqdm

from FriendRecognize.utils.Filtering import preprocessing
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


def load_images_from(path_source, get_images=True, get_image_name=True, show_tqdm=True):
    images = []
    if show_tqdm:
        for image_name in tqdm(os.listdir(path_source), "Load images from: " + path_source):
            image = cv.imread(os.path.join(path_source, image_name))
            if image is not None:
                if get_images and get_image_name:
                    images.append((image, image_name))
                elif get_images:
                    images.append(image)
                elif get_image_name:
                    images.append(image_name)
                else:
                    raise
        return images
    else:
        for image_name in os.listdir(path_source):
            image = cv.imread(os.path.join(path_source, image_name))
            if image is not None:
                if get_images and get_image_name:
                    images.append((image, image_name))
                elif get_images:
                    images.append(image)
                elif get_image_name:
                    images.append(image_name)
                else:
                    raise
        return images


# LBP
def extraction_feature_LBP(X, kind_of_feature):
    lbp = LBP(numPoints=16, radius=4, num_bins=100, n_row=10, n_col=10)
    x_new = []
    for i in tqdm(range(X.shape[0]), desc=kind_of_feature):
        hist, _ = lbp.describe(X[i])
        x_new.append(hist)
    x_new = np.array(x_new)
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
