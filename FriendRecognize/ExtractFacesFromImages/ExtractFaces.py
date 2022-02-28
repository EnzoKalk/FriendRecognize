import os

import cv2 as cv
import dlib
from FaceAligner import FaceAligner
from tqdm import tqdm


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


if __name__ == '__main__':
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    faceNet = cv.dnn.readNet(faceModel, faceProto)
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=224, desiredFaceHeight=224)

    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    source_images = "./source"

    if not os.path.exists(source_images):
        print("There are no images to detect")
        exit(1)

    destination_images = "./destination"
    if not os.path.exists(destination_images):
        os.makedirs(destination_images)

    padding = 20
    i = 0
    for image in tqdm(load_images_from_folder(source_images), "Detect faces from images"):
        scale_factor = 2
        new_height = int(image.shape[0] * scale_factor)
        new_width = int(image.shape[1] * scale_factor)
        dimensions = (new_width, new_height)
        image = cv.resize(image, dimensions, interpolation=cv.INTER_LANCZOS4)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv.imwrite(destination_images + "/" + str(i) + ".jpg",
                       image[y - padding:y + h + padding, x - padding:x + w + padding])
            i += 1
