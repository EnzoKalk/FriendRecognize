import copy
import cv2 as cv
import math
import numpy as np


def is_colored(image):
    return len(image.shape) == 3


def remove_color_from(image):
    try:
        if is_colored(image):
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return image
    except:
        return None


def denoise_preserving_edge_from(image, epsilon=0.01, neighborhood_pixels=3):
    try:
        if is_colored(image):
            cv.ximgproc.guidedFilter(image, image, neighborhood_pixels, epsilon, image)
        else:
            cv.ximgproc.guidedFilter(image, image, neighborhood_pixels, epsilon, image, -1)
        return image
    except:
        return None


def get_landmarks_from(image, detector, predictor):
    faces = detector(image, 1)
    for rect in faces:
        return predictor(image, rect)


def align_face_from(image, detector, predictor):
    try:
        shape = get_landmarks_from(image, detector, predictor)
        x_sx = (shape.part(36).x + shape.part(37).x + shape.part(38).x +
                shape.part(39).x + shape.part(40).x + shape.part(41).x) / 6
        y_sx = (shape.part(36).y + shape.part(37).y + shape.part(38).y +
                shape.part(39).y + shape.part(40).y + shape.part(41).y) / 6
        x_dx = (shape.part(42).x + shape.part(43).x + shape.part(44).x +
                shape.part(45).x + shape.part(46).x + shape.part(47).x) / 6
        y_dx = (shape.part(42).y + shape.part(43).y + shape.part(44).y +
                shape.part(45).y + shape.part(46).y + shape.part(47).y) / 6
        angular_coefficient = math.atan((y_dx - y_sx) / (x_dx - x_sx))
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotate_matrix = cv.getRotationMatrix2D(center=center, angle=math.degrees(angular_coefficient), scale=1)
        return cv.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
    except:
        return None


def crop_by_landmarks(image, landmarks, range_face, landmark_y_min, landmark_y_max):
    try:
        under_face_bridge_x = []
        under_face_bridge_y = []
        for i in range_face:
            under_face_bridge_x.append(landmarks[i][0])
            under_face_bridge_y.append(landmarks[i][1])
        x_min = min(under_face_bridge_x)
        x_max = max(under_face_bridge_x)
        y_min = landmarks[landmark_y_min][1]
        y_max = landmarks[landmark_y_max][1]
        return image[y_min:y_max, x_min:x_max]
    except:
        return None


def crop_face_from(image, detector, predictor):
    try:
        landmarks = np.array(
            [[coordinate.x, coordinate.y] for coordinate in predictor(image, detector(image)[0]).parts()])
        range_face = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        landmark_y_min = 19
        landmark_y_max = 8
        return crop_by_landmarks(image, landmarks, range_face, landmark_y_min, landmark_y_max)
    except:
        return None


def highlight_a_face_from(image, size):
    try:
        image = cv.resize(image, (size, size), cv.INTER_LANCZOS4)
        if image is not None:
            image = cv.equalizeHist(image)
        return image
    except:
        return None


def preprocessing(image, detector, predictor, size=200):
    image = remove_color_from(image)
    recovery_image = copy.deepcopy(image)
    if image is not None:
        image = denoise_preserving_edge_from(image)
    if image is not None:
        image = align_face_from(image, detector, predictor)
    if image is not None:
        image = crop_face_from(image, detector, predictor)
    if image is None:
        image = recovery_image
    return highlight_a_face_from(image, size)
