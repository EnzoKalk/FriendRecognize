import os
import shutil

import cv2 as cv
import dlib
import yaml
from tqdm import tqdm

from FriendRecognize.utils.Filtering import align_face_from, crop_face_from
from FriendRecognize.utils.UsefulMethods import load_images_from


def get_face_cascade_path(config):
    return config['libs']['face_cascade']


def get_predictor_path(config):
    return config['libs']['predictor']


def get_source_path(config):
    return config['data']['source_images']


def get_destination_path(config):
    return config['data']['destination_extracted_faces']


def extract_face_from(image, destination_path, face_cascade, image_scale_factor=2, padding=50):
    image, image_name = image
    try:
        new_height = int(image.shape[0] * image_scale_factor)
        new_width = int(image.shape[1] * image_scale_factor)
        image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_LANCZOS4)
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image,
                                              scaleFactor=1.1,
                                              minNeighbors=5,
                                              minSize=(50, 50),
                                              flags=cv.CASCADE_SCALE_IMAGE)
        num_face = 0
        for (x, y, width, height) in faces:
            cv.imwrite(os.path.join(destination_path + "/temp", "face" + str(num_face) + "_" + image_name),
                       image[y - padding:y + height + padding, x - padding:x + width + padding])
            num_face += 1
    except:
        pass


def align_and_crop_faces_from(face, detector, predictor):
    if face is not None:
        face = align_face_from(face, detector, predictor)
    if face is not None:
        face = crop_face_from(face, detector, predictor)
        return face
    return None


if __name__ == '__main__':
    # Init parameters
    with open('config.yml') as file:
        config = yaml.full_load(file)
    face_cascade = cv.CascadeClassifier(get_face_cascade_path(config))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(get_predictor_path(config))
    source_path = get_source_path(config)
    destination_path = get_destination_path(config)

    # Check directories
    if not os.path.exists(source_path):
        print("There are no images for detecting faces")
        exit(1)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    if not os.path.exists(destination_path + "/temp"):
        os.makedirs(destination_path + "/temp")

    # Detect faces
    for image in tqdm(load_images_from(source_path), "Detect faces"):
        extract_face_from(image, destination_path, face_cascade)
    # Crop and align faces
    for image in tqdm(load_images_from(destination_path + "/temp"), "Align and crop faces"):
        face, image_name = image
        face = align_and_crop_faces_from(face, detector, predictor)
        # Store face
        try:
            cv.imwrite(os.path.join(destination_path, image_name), face)
        except:
            pass

    # Delete temp folder
    shutil.rmtree(destination_path + "/temp")

    exit(0)
