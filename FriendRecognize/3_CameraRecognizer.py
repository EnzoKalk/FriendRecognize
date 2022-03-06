import cv2 as cv
import dlib
import numpy as np
import yaml

from FriendRecognize.utils.Filtering import preprocessing
from FriendRecognize.utils.UsefulMethods import extraction_feature_LBP
from utils.trainModelAndClassifier.Classifier import Classifier


def get_model(config, feature):
    return config['models'][feature]


def get_friends(config):
    return config['friends']['with']


def get_predictor_path(config):
    return config['libs']['predictor']


def get_face_cascade_path(config):
    return config['libs']['face_cascade']


def draw_detection(x, y, h, w, color_rgb, label, background_label=(255, 255, 255), thickness=2, font_scale=0.6):
    cv.rectangle(frames, (x, y), (x + w, y + h), color_rgb, thickness)
    cv.rectangle(frames, (x, y), (x + w, y - 30), background_label, -1)
    cv.putText(frames,
               label,
               (x, y - 10),
               cv.FONT_HERSHEY_SIMPLEX,
               font_scale,
               color_rgb,
               thickness)


def compute_feature_for_testing(X_val):
    X_val = extraction_feature_LBP(X_val, "Extract feature for prediction")
    return X_val


def extract_features_for_testing(img, detector, predictor):
    prep_img = preprocessing(img, detector, predictor)
    if prep_img is None:
        return []
    X = [prep_img]
    return compute_feature_for_testing(np.array(X))


if __name__ == '__main__':
    # Init parameters
    with open('config.yml') as file:
        config = yaml.full_load(file)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(get_predictor_path(config))
    face_cascade = cv.CascadeClassifier(get_face_cascade_path(config))
    models = []
    for friend in get_friends(config):
        model = Classifier()
        model.load(get_model(config, friend))
        models.append(model)
    padding = 50

    # Check faces
    video_capture = cv.VideoCapture(0)
    while True:
        # Detect face
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
            # Extract feature from face
            X = extract_features_for_testing(frames[y - padding:y + h + padding, x - padding:x + w + padding],
                                             detector,
                                             predictor)
            # Predict probabilities
            classifier_probabilities = []
            if X is not None and hasattr(X, 'shape'):
                if len(X.shape) == 2:
                    i = 0
                    max_probability = 0
                    for i in range(len(get_friends(config))):
                        probability = round(models[i].predict_proba(X)[0][1] * 100, 3)
                        classifier_probabilities.append(probability)
                        if probability > max_probability:
                            max_probability = probability
                else:
                    for friend in get_friends(config):
                        classifier_probabilities.append(0)
                    max_probability = 0
            else:
                for friend in get_friends(config):
                    classifier_probabilities.append(0)
                max_probability = 0

            # Draw rectangle and labels
            if max_probability < 50.0:
                draw_detection(x, y, h, w,
                               (169, 169, 169),
                               "Sconosciuto")
            elif classifier_probabilities[0] >= max(classifier_probabilities[1],
                                                    classifier_probabilities[2],
                                                    classifier_probabilities[3],
                                                    classifier_probabilities[4]):
                draw_detection(x, y, h, w,
                               (255, 0, 0),
                               "Vincenzo " + str(round(classifier_probabilities[0], 1)) + "%")

            elif classifier_probabilities[1] >= max(classifier_probabilities[0],
                                                    classifier_probabilities[2],
                                                    classifier_probabilities[3],
                                                    classifier_probabilities[4]):
                draw_detection(x, y, h, w,
                               (124, 252, 0),
                               "Angelo " + str(round(classifier_probabilities[1], 1)) + "%")
            elif classifier_probabilities[2] >= max(classifier_probabilities[0],
                                                    classifier_probabilities[1],
                                                    classifier_probabilities[3],
                                                    classifier_probabilities[4]):
                draw_detection(x, y, h, w,
                               (255, 123, 25),
                               "Dima " + str(round(classifier_probabilities[2], 1)) + "%")
            elif classifier_probabilities[3] >= max(classifier_probabilities[0],
                                                    classifier_probabilities[1],
                                                    classifier_probabilities[2],
                                                    classifier_probabilities[4]):
                draw_detection(x, y, h, w,
                               (255, 255, 0),
                               "Giovanna " + str(round(classifier_probabilities[3], 1)) + "%")
            elif classifier_probabilities[4] >= max(classifier_probabilities[0],
                                                    classifier_probabilities[1],
                                                    classifier_probabilities[2],
                                                    classifier_probabilities[3]):
                draw_detection(x, y, h, w,
                               (128, 0, 128),
                               "Noemi " + str(round(classifier_probabilities[4], 1)) + "%")
        # Update frame
        cv.imshow('Video', frames)

        # Stop program
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
