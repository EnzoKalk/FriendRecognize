# Import required modules
import os

import cv2 as cv
# Load detector
from tqdm import tqdm


def load_images_from_folder(folder, cv=None):
    images = []
    for filename in tqdm(os.listdir(folder), "Read Images"):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def get_lendmarks_from(image, detector, predictor):
    grayframe = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = detector(grayframe, 1)
    showface = image.copy()
    for rect in faces:
        shape = predictor(grayframe, rect)
        c = (0, 255, 0)
        for i in range(0, 68):
            p = (shape.part(i).x, shape.part(i).y)
            cv.circle(showface, p, 2, c, -1)
    return shape
