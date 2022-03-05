import cv2 as cv
import dlib
import numpy as np
import yaml

from FriendRecognize.utils.Filtering import preprocessing
from FriendRecognize.utils.UsefulMethods import extraction_feature_LBP
from utils.trainModelAndClassifier.Classifier import Classifier


def get_model(config, feature):
    return config['models'][feature]


def get_features(config):
    return config['features']['with']


def get_predictor_path(config):
    return config['libs']['predictor']


def get_face_cascade_path(config):
    return config['libs']['face_cascade']


def compute_feature_for_testing(X_val):
    X_val = extraction_feature_LBP(X_val, "")
    return X_val


def extract_features_for_testing(img, detector, predictor):
    prep_img = preprocessing(img, detector, predictor)
    if prep_img is None:
        return []
    X = []
    X.append(prep_img)
    return compute_feature_for_testing(np.array(X))


if __name__ == '__main__':
    # Init parameters
    with open('config.yml') as file:
        config = yaml.full_load(file)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(get_predictor_path(config))
    face_cascade = cv.CascadeClassifier(get_face_cascade_path(config))
    models = []
    for feature in get_features(config):
        model = Classifier()
        model.load(get_model(config, feature))
        models.append(model)
    padding = 50

    video_capture = cv.VideoCapture(0)
    while True:
        ret, frames = video_capture.read()
        gray = cv.cvtColor(frames, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            X = extract_features_for_testing(frames[y - padding:y + h + padding, x - padding:x + w + padding],
                                             detector,
                                             predictor)
            classifier_probabilities = []
            if X is not None and hasattr(X, 'shape'):
                if len(X.shape) == 2:
                    i = 0
                    max_probability = 0
                    for i in range(len(get_features(config))):
                        probability = round(models[i].predict_proba(X)[0][1], 3)
                        classifier_probabilities.append(probability)
                        if probability > max_probability:
                            max_probability = probability
                else:
                    for feature in get_features(config):
                        classifier_probabilities.append(0)
                    max_probability = 0
            else:
                for feature in get_features(config):
                    classifier_probabilities.append(0)
                max_probability = 0

            if max_probability < 0.5:
                cv.rectangle(frames, (x, y), (x + w, y + h), (169, 169, 169), 2)
                cv.rectangle(frames, (x, y), (x + w, y - 30), (255, 255, 255), -1)
                cv.putText(frames,
                           "Sconosciuto",
                           (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (169, 169, 169),
                           2)
            elif classifier_probabilities[0] >= max(classifier_probabilities[1],
                                                    classifier_probabilities[2],
                                                    classifier_probabilities[3],
                                                    classifier_probabilities[4]):
                cv.rectangle(frames, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv.rectangle(frames, (x, y), (x + w, y - 30), (255, 255, 255), -1)
                cv.putText(frames,
                           "Vincenzo " + str(round(classifier_probabilities[0] * 100, 1)) + "%",
                           (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (255, 0, 0),
                           2)
            elif classifier_probabilities[1] >= max(classifier_probabilities[0],
                                                    classifier_probabilities[2],
                                                    classifier_probabilities[3],
                                                    classifier_probabilities[4]):
                cv.rectangle(frames, (x, y), (x + w, y + h), (124, 252, 0), 2)
                cv.rectangle(frames, (x, y), (x + w, y - 30), (255, 255, 255), -1)
                cv.putText(frames,
                           "Angelo " + str(round(classifier_probabilities[1] * 100, 1)) + "%",
                           (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (124, 252, 0),
                           2)
            elif classifier_probabilities[2] >= max(classifier_probabilities[0],
                                                    classifier_probabilities[1],
                                                    classifier_probabilities[3],
                                                    classifier_probabilities[4]):
                cv.rectangle(frames, (x, y), (x + w, y + h), (255, 123, 25), 2)
                cv.rectangle(frames, (x, y), (x + w, y - 30), (255, 255, 255), -1)
                cv.putText(frames,
                           "Dima " + str(round(classifier_probabilities[2] * 100, 1)) + "%",
                           (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (255, 123, 25),
                           2)
            elif classifier_probabilities[3] >= max(classifier_probabilities[0],
                                                    classifier_probabilities[1],
                                                    classifier_probabilities[2],
                                                    classifier_probabilities[4]):
                cv.rectangle(frames, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv.rectangle(frames, (x, y), (x + w, y - 30), (255, 255, 255), -1)
                cv.putText(frames,
                           "Giovanna " + str(round(classifier_probabilities[3] * 100, 1)) + "%",
                           (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (255, 255, 0),
                           2)
            elif classifier_probabilities[4] >= max(classifier_probabilities[0],
                                                    classifier_probabilities[1],
                                                    classifier_probabilities[2],
                                                    classifier_probabilities[3]):
                cv.rectangle(frames, (x, y), (x + w, y + h), (128, 0, 128), 2)
                cv.rectangle(frames, (x, y), (x + w, y - 30), (255, 255, 255), -1)
                cv.putText(frames,
                           "Noemi " + str(round(classifier_probabilities[4] * 100, 1)) + "%",
                           (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (128, 0, 128),
                           2)
        cv.imshow('Video', frames)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
