import cv2 as cv
import numpy as np
import os
import pickle as pk
from enum import Enum
from tqdm import tqdm

from FriendRecognize.utils.lbp.LBPFeature import LBP


class Friend(Enum):
    VINCENZO = "Vincenzo"
    ANGELO = "Angelo"
    DIMA = "Dima"
    GIOVANNA = "Giovanna"
    NOEMI = "Noemi"


class ImageType(Enum):
    FRIEND = 1
    NO_FRIEND = 0


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


def extraction_feature_LBP(X, kind_of_feature):
    lbp = LBP(numPoints=16, radius=4, num_bins=100, n_row=10, n_col=10)
    x_new = []
    for i in tqdm(range(X.shape[0]), desc=kind_of_feature):
        hist, _ = lbp.describe(X[i])
        x_new.append(hist)
    x_new = np.array(x_new)
    return x_new
