import os
import random

import cv2 as cv
import numpy as np
from tqdm import tqdm

from FriendRecognize.utils.UsefulMethods import load_images_from


def previous_augmentation(path):
    for image in load_images_from(path, show_tqdm=False):
        image, image_name = image
        img = cv.flip(image, 1)
        cv.imwrite(path + "/" + "flipped_" + image_name, img)


def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)


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


def further_augmentation(path, remaining):
    images = load_images_from(path, show_tqdm=False)
    while remaining > 0:
        image, image_name = random.choice(images)
        n = random.randint(0, 3)
        if n == 0:
            img = adjust_gamma(image, round(random.uniform(1.5, 1.9), 1))
        elif n == 1:
            img = adjust_gamma(image, round(random.uniform(0.1, 0.5), 1))
        elif n == 2:
            img = noisy("gauss", image)
        else:
            img = noisy("poisson", image)
        cv.imwrite(path + "/" + str(remaining) + "_" + image_name, img)
        remaining -= 1


def augment(paths):
    max_number_of_images = 0
    for path in tqdm(paths, "Add flip images for each path"):
        previous_augmentation(path)
        number_of_images = len([name for name in (os.listdir(path))])
        if number_of_images > max_number_of_images:
            max_number_of_images = number_of_images
    for path in tqdm(paths, "Add random noise or gamma correction for each path"):
        remaining = max_number_of_images - len([name for name in (os.listdir(path))])
        further_augmentation(path, remaining)
