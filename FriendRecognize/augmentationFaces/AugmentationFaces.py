import os
import random
import sys

import cv2 as cv
import dlib
import numpy as np
import yaml
from tqdm import tqdm


def get_predictor_path(config):
    return "../" + config['libs']['predictor']

def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
    else:
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
    return noisy

def previous_augmentation(path):
    for image in tqdm(load_images_from_folder(path), desc="Flip images"):
        im = image[1]
        img = cv.flip(im, 1)
        cv.imwrite(path + "/" + image[0] + "_flipped.jpg", img)

def augmentation(path, remaining):
    images = load_images_from_folder(path)
    while remaining > 0:
        image = random.choice(images)
        im = image[1]
        n = random.randint(0, 3)
        if n == 0:
            img = adjust_gamma(im, round(random.uniform(1.3, 1.9), 1))
        elif n == 1:
            img = adjust_gamma(im, round(random.uniform(0.1, 0.7), 1))
        elif n == 2:
            img = noisy("gauss", im)
        else:
            img = noisy("poisson", im)

        cv.imwrite(path + "/" + image[0] + "_" + str(remaining) + ".jpg", img)
        remaining -= 1

def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        images.append((filename, img))
    return images


if __name__ == "__main__":
    # Paths
    path_vincenzo = '../data/trainingSet/Vincenzo'
    path_angelo = '../data/trainingSet/Angelo'
    path_dima = '../data/trainingSet/Dima'
    path_giovanna = '../data/trainingSet/Giovanna'
    path_noemi = '../data/trainingSet/Noemi'

    previous_augmentation(path_vincenzo)
    previous_augmentation(path_angelo)
    previous_augmentation(path_dima)
    previous_augmentation(path_giovanna)
    previous_augmentation(path_noemi)

    with open('../config.yml') as file:
        config = yaml.full_load(file)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(get_predictor_path(config))

    number_vincenzo_elements = len([name for name in tqdm(os.listdir(path_vincenzo), desc="Count Vincenzo elements")])
    number_angelo_elements = len([name for name in tqdm(os.listdir(path_angelo), desc="Count Angelo elements")])
    number_dima_elements = len([name for name in tqdm(os.listdir(path_dima), desc="Count Dima elements")])
    number_giovanna_elements = len([name for name in tqdm(os.listdir(path_giovanna), desc="Count Giovanna elements")])
    number_noemi_elements = len([name for name in tqdm(os.listdir(path_noemi), desc="Count Noemi elements")])

    max = max(number_vincenzo_elements,
              number_angelo_elements,
              number_dima_elements,
              number_giovanna_elements,
              number_noemi_elements)

    remaining_vincenzo = max - number_vincenzo_elements
    remaining_angelo = max - number_angelo_elements
    remaining_dima = max - number_dima_elements
    remaining_giovanna = max - number_giovanna_elements
    remaining_noemi = max - number_noemi_elements

    augmentation(path_vincenzo, remaining_vincenzo)
    augmentation(path_angelo, remaining_angelo)
    augmentation(path_dima, remaining_dima)
    augmentation(path_giovanna, remaining_giovanna)
    augmentation(path_noemi, remaining_noemi)

    sys.exit(0)
