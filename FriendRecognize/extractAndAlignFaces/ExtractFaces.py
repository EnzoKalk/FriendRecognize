import math
import os

import cv2 as cv
import numpy as np
from tqdm import tqdm

from FriendRecognize.utils.CropAndFilter import get_lendmarks_from, crop_by_landmarks


def load_images_from(path_source):
    images = []
    for image_name in tqdm(os.listdir(path_source), "Load images from: " + path_source):
        image = cv.imread(os.path.join(path_source, image_name))
        if image is not None:
            images.append((image, image_name))
    return images

def align_face_from(extracted_faces, path_destination, detector, predictor, ):

    for image in tqdm(extracted_faces, "Align faces"):
        image, image_name = image
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
            cv.imwrite(os.path.join(path_destination, image_name), image)
        except:
            pass


def extract_face_from(image, face_cascade, image_scale_factor, padding, extracted_faces):
    image, image_name = image
    try:
        new_height = int(image.shape[0] * image_scale_factor)
        new_width = int(image.shape[1] * image_scale_factor)
        image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_LANCZOS4)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,
                                              scaleFactor=1.1,
                                              minNeighbors=5,
                                              minSize=(50, 50),
                                              flags=cv.CASCADE_SCALE_IMAGE)
        for (x, y, width, height) in faces:
            extracted_faces.append((image[y - padding:y + height + padding, x - padding:x + width + padding], image_name))
    except Exception as e:
        print("Some problems to extract face from:" + image_name + ".jpg (" + str(e) + ")")


def extract_and_align_faces_from(path_source, path_destination, detector, predictor, face_cascade, image_scale_factor=2, padding=50,):
    extracted_faces = []

    if not os.path.exists(path_source):
        print("There are no images for detecting faces")
        exit(1)
    if not os.path.exists(path_destination):
        os.makedirs(path_destination)

    for image in tqdm(load_images_from(path_source), "Detect faces from images"):
        extract_face_from(image, face_cascade, image_scale_factor, padding, extracted_faces)
        align_face_from(extracted_faces, path_destination, detector, predictor)
