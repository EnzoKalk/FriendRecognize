import os
import sys

import cv2 as cv
import dlib
import yaml
from tqdm import tqdm


def get_predictor_path(config):
    return "../" + config['libs']['predictor']


def augmentation(path, remaining):
    for image in tqdm(load_images_from_folder(path), "Augmentation Flip to: " + path):
        if remaining == 0:
            break
        img = cv.flip(image[1], 1)
        cv.imwrite(path + "/" + image[0] + "_" + str(remaining) + ".jpg", img)
        remaining -= 1


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        images.append((filename[:-4] + "_aug_glasses", img))
    return images


if __name__ == "__main__":
    # Paths
    path_vincenzo = '../data/trainingSet/Vincenzo'
    path_angelo = '../data/trainingSet/Angelo'
    path_dima = '../data/trainingSet/Dima'
    path_giovanna = '../data/trainingSet/Giovanna'
    path_noemi = '../data/trainingSet/Noemi'

    # Init params
    with open('../config.yml') as file:
        config = yaml.full_load(file)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(get_predictor_path(config))

    number_vincenzo_elements = len([name for name in tqdm(os.listdir(path_vincenzo), desc="Count Vincenzo elements")])
    number_angelo_elements = len([name for name in tqdm(os.listdir(path_angelo), desc="Count Angelo elements")])
    number_dima_elements = len([name for name in tqdm(os.listdir(path_dima), desc="Count Dima elements")])
    number_giovanna_elements = len([name for name in tqdm(os.listdir(path_noemi), desc="Count Giovanna elements")])
    number_noemi_elements = len([name for name in tqdm(os.listdir(path_giovanna), desc="Count Noemi elements")])

    # Calculate number of element for the augmentation
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

    # Exit
    sys.exit(0)
