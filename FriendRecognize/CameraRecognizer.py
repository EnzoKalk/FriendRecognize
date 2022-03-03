import os

import cv2 as cv
import dlib
import numpy as np

from FriendRecognize.utils.CropAndFilter import preprocessing
from utils.FaceAligner import FaceAligner
from utils.classifier.FeatureClassifier import FeatureClassifier
from utils.lbp.LBPFeature import LBP


def extraction_feature_LBP(X):
    lbp = LBP(numPoints=16, radius=4, num_bins=100, n_row=10, n_col=10)
    x_new = []
    for i in (X):
        hist, _ = lbp.describe(i)
        x_new.append(hist)
    x_new = np.array(x_new)
    return x_new


def compute_feature_for_testing(X_val):
    X_val = extraction_feature_LBP(X_val)
    return X_val


def extract_features_for_testing(img, detector, predictor):
    prep_img = preprocessing(img, detector, predictor)
    if prep_img is None:
        return []
    X = []
    X.append(prep_img)
    return compute_feature_for_testing(np.array(X))


if __name__ == '__main__':
    model_vincenzo = FeatureClassifier()
    model_vincenzo.load(os.path.join('models', 'Vincenzo'))
    model_angelo = FeatureClassifier()
    model_angelo.load(os.path.join('models', 'Angelo'))
    model_dima = FeatureClassifier()
    model_dima.load(os.path.join('models', 'Dima'))
    model_giovanna = FeatureClassifier()
    model_giovanna.load(os.path.join('models', 'Giovanna'))
    model_noemi = FeatureClassifier()
    model_noemi.load(os.path.join('models', 'Noemi'))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('libs/shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=224, desiredFaceHeight=224)
    face_cascade = cv.CascadeClassifier('libs/haarcascade_frontalface_default.xml')
    video_capture = cv.VideoCapture(0)
    padding = 0
    while True:
        ret, frames = video_capture.read()
        gray = cv.cvtColor(frames, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(150, 150),
            flags=cv.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            image = frames[y - padding:y + h + padding, x - padding:x + w + padding]
            X = extract_features_for_testing(image, detector, predictor)
            if X is not None and hasattr(X, 'shape'):
                if len(X.shape) == 2:
                    classifier_probability_vincenzo = round(model_vincenzo.predict_proba(X)[0][1], 3)
                    classifier_probability_angelo = round(model_angelo.predict_proba(X)[0][1], 3)
                    classifier_probability_dima = round(model_dima.predict_proba(X)[0][1], 3)
                    classifier_probability_giovanna = round(model_giovanna.predict_proba(X)[0][1], 3)
                    classifier_probability_noemi = round(model_noemi.predict_proba(X)[0][1], 3)
                    max_probability = max(classifier_probability_vincenzo,
                                          classifier_probability_angelo,
                                          classifier_probability_dima,
                                          classifier_probability_giovanna,
                                          classifier_probability_noemi)
                else:
                    classifier_probability_vincenzo = [[0, 0]]
                    classifier_probability_angelo = [[0, 0]]
                    classifier_probability_dima = [[0, 0]]
                    classifier_probability_giovanna = [[0, 0]]
                    classifier_probability_noemi = [[0, 0]]
                    max_probability = 0
            else:
                classifier_probability_vincenzo = [[0, 0]]
                classifier_probability_angelo = [[0, 0]]
                classifier_probability_dima = [[0, 0]]
                classifier_probability_giovanna = [[0, 0]]
                classifier_probability_noemi = [[0, 0]]
                max_probability = 0
            if 0.5 >= max_probability:
                cv.rectangle(frames, (x, y), (x + w, y + h), (169, 169, 169), 2)
                cv.rectangle(frames, (x, y), (x + w, y - 30), (255, 255, 255), -1)
                cv.putText(frames,
                           "Sconosciuto",
                           (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (169, 169, 169),
                           2)
            elif classifier_probability_vincenzo >= max(classifier_probability_angelo,
                                                        classifier_probability_dima,
                                                        classifier_probability_giovanna,
                                                        classifier_probability_noemi):
                cv.rectangle(frames, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv.rectangle(frames, (x, y), (x + w, y - 30), (255, 255, 255), -1)
                cv.putText(frames,
                           "Vincenzo " + str(round(classifier_probability_vincenzo * 100, 1)) + "%",
                           (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (255, 0, 0),
                           2)
            elif classifier_probability_angelo >= max(classifier_probability_vincenzo,
                                                      classifier_probability_dima,
                                                      classifier_probability_giovanna,
                                                      classifier_probability_noemi):
                cv.rectangle(frames, (x, y), (x + w, y + h), (124, 252, 0), 2)
                cv.rectangle(frames, (x, y), (x + w, y - 30), (255, 255, 255), -1)
                cv.putText(frames,
                           "Angelo " + str(round(classifier_probability_angelo * 100, 1)) + "%",
                           (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (124, 252, 0),
                           2)
            elif classifier_probability_dima >= max(classifier_probability_vincenzo,
                                                    classifier_probability_angelo,
                                                    classifier_probability_giovanna,
                                                    classifier_probability_noemi):
                cv.rectangle(frames, (x, y), (x + w, y + h), (255, 123, 25), 2)
                cv.rectangle(frames, (x, y), (x + w, y - 30), (255, 255, 255), -1)
                cv.putText(frames,
                           "Dima " + str(round(classifier_probability_dima * 100, 1)) + "%",
                           (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (255, 123, 25),
                           2)
            elif classifier_probability_giovanna >= max(classifier_probability_vincenzo,
                                                        classifier_probability_angelo,
                                                        classifier_probability_dima,
                                                        classifier_probability_noemi):
                cv.rectangle(frames, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv.rectangle(frames, (x, y), (x + w, y - 30), (255, 255, 255), -1)
                cv.putText(frames,
                           "Giovanna " + str(round(classifier_probability_giovanna * 100, 1)) + "%",
                           (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (255, 255, 0),
                           2)
            elif classifier_probability_noemi >= max(classifier_probability_vincenzo,
                                                     classifier_probability_angelo,
                                                     classifier_probability_dima,
                                                     classifier_probability_giovanna):
                cv.rectangle(frames, (x, y), (x + w, y + h), (128, 0, 128), 2)
                cv.rectangle(frames, (x, y), (x + w, y - 30), (255, 255, 255), -1)
                cv.putText(frames,
                           "Noemi " + str(round(classifier_probability_noemi * 100, 1)) + "%",
                           (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (128, 0, 128),
                           2)
        cv.imshow('Video', frames)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
