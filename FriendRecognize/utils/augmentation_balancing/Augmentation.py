import cv2 as cv
import numpy as np
import os
import random
from tqdm import tqdm

from FriendRecognize.utils.UsefulMethods import load_images_from


def previous_augmentation(path):
    for image in load_images_from(path, show_tqdm=False):
        # Init parameters
        image, image_name = image

        # Augmentation
        image = cv.flip(image, 1)

        # Store
        cv.imwrite(os.path.join(path, "flipped_" + image_name), image)


def adjust_gamma(image, gamma):
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)


def noisy(noise_typ, image):
    # Gaussian
    if noise_typ == "gauss":
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape)
        gauss = gauss.reshape(image.shape)
        return image + gauss
    else:  # Poisson
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        return np.random.poisson(image * vals) / float(vals)


def further_augmentation(path, remaining):
    # Init parameters
    images = load_images_from(path, show_tqdm=False)

    # Augmentation
    while remaining > 0:
        # Init parameters
        image, image_name = random.choice(images)
        random_choice = random.randint(0, 3)
        remaining -= 1

        # Select choice
        if random_choice == 0:
            image = adjust_gamma(image, round(random.uniform(1.5, 1.9), 1))
        elif random_choice == 1:
            image = adjust_gamma(image, round(random.uniform(0.1, 0.5), 1))
        elif random_choice == 2:
            image = noisy("gauss", image)
        else:
            image = noisy("poisson", image)

        # Store
        cv.imwrite(os.path.join(path, str(remaining) + "_" + image_name), image)


def augmentation(paths):
    # Init prameters
    max_number_of_images = 0

    # Add flipped images
    for path in tqdm(paths, "Add flip images for each path"):
        previous_augmentation(path)
        number_of_images = len([name for name in (os.listdir(path))])
        if number_of_images > max_number_of_images:
            max_number_of_images = number_of_images

    # Add noised or gamma corrected images
    for path in tqdm(paths, "Add noised or gamma corrected images for each path"):
        remaining = max_number_of_images - len([name for name in (os.listdir(path))])
        further_augmentation(path, remaining)
