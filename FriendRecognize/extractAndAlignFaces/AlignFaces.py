import math
import os

import cv2 as cv
import dlib
import numpy as np
from tqdm import tqdm

from FriendRecognize.utils.CropAndFilter import crop_by_landmarks
from FriendRecognize.utils.FaceLandmarks import get_lendmarks_from


def load_images_from_folder(folder):
    images = []
    for filename in tqdm(os.listdir(folder), "Read Images"):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((img, filename))
    return images


if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../libs/shape_predictor_68_face_landmarks.dat')
    source_images = "./datasetDetectedFaces/extracted_faces"
    destination_images = "./datasetDetectedFaces/aligned_faces"
    if not os.path.exists(destination_images):
        os.makedirs(destination_images)
    for img in tqdm(load_images_from_folder(source_images), "Find lendmarks on faces"):
        image = img[0]
        image_name = img[1]
        try:
            shape = get_lendmarks_from(image, detector, predictor)
            x_sx = (shape.part(36).x +
                    shape.part(37).x +
                    shape.part(38).x +
                    shape.part(39).x +
                    shape.part(40).x +
                    shape.part(41).x) / 6
            y_sx = (shape.part(36).y +
                    shape.part(37).y +
                    shape.part(38).y +
                    shape.part(39).y +
                    shape.part(40).y +
                    shape.part(41).y) / 6
            x_dx = (shape.part(42).x +
                    shape.part(43).x +
                    shape.part(44).x +
                    shape.part(45).x +
                    shape.part(46).x +
                    shape.part(47).x) / 6
            y_dx = (shape.part(42).y +
                    shape.part(43).y +
                    shape.part(44).y +
                    shape.part(45).y +
                    shape.part(46).y +
                    shape.part(47).y) / 6
            angular_coefficient = math.atan((y_dx - y_sx) / (x_dx - x_sx))
            height, width = image.shape[:2]
            center = (width / 2, height / 2)
            rotate_matrix = cv.getRotationMatrix2D(center=center, angle=math.degrees(angular_coefficient), scale=1)
            image = cv.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
            rect = detector(image)[0]
            sp = predictor(image, rect)
            landmarks = np.array([[p.x, p.y] for p in sp.parts()])
            range_face = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            landmark_y_min = 19
            landmark_y_max = 8
            image = crop_by_landmarks(image, landmarks, range_face, landmark_y_min, landmark_y_max)
            cv.imwrite(os.path.join(destination_images, image_name), image)
        except:
            pass
