import copy
import os
import random

import cv2 as cv
import numpy as np
from tqdm import tqdm

from FriendRecognize.utils.Filtering import preprocessing
from FriendRecognize.utils.UsefulMethods import ImageType, Labeler, Friend, extraction_feature_LBP
from FriendRecognize.utils.object.TrainImage import TrainImage
from FriendRecognize.utils.training_model.Classifier import Classifier, print_metrics


def get_model(config, feature):
    return config['models'][feature]


def get_image_with_friend(config, feature):
    path = config['data']['training'] + config['friends']['with'][feature]
    img_type = ImageType.FRIEND.value
    return {'path': path, 'type': img_type}


def get_image_without_friend(config, feature):
    path = config['data']['training'] + config['friends']['without'][feature]
    img_type = ImageType.NO_FRIEND.value
    return {'path': path, 'type': img_type}


def generate_empty_dataset():
    return {'train': {0: [], 1: []}, 'val': {0: [], 1: []}}


def compute_raw_feature_for_training(X_train, y_train, X_val, y_val, friend):
    X_train = extraction_feature_LBP(X_train,
                                     "Extract feature: trainingSet-set of " + str(friend))
    X_val = extraction_feature_LBP(X_val,
                                   "Extract feature: validation-set of " + str(friend))
    return X_train, y_train, X_val, y_val


def load_raw_feature_for_training(set_files, labeler, detector, predictor, friend):
    X = []
    y = []
    images = []
    for id in tqdm(set_files, desc="Loaded identities (" + str(friend) + ")"):
        for file in set_files[id]:
            images.append(cv.imread(file.path))
            y.append(labeler.encode(id))
    for img in tqdm(images, desc="Pre-processing " + str(friend)):
        image_to_filter = copy.deepcopy(img)
        image = preprocessing(image_to_filter, detector, predictor)
        if image is not None:
            X.append(image)
        else:
            del y[y[len(X)]]
    return np.array(X), np.array(y)


def extract_features_for_training(dataset, labeler, detector, predictor, friend):
    X = {}
    y = {}
    for dataset_type in ['train', 'val']:
        print(f"\n ---> Pre-processing {dataset_type} set <--- ")
        X[dataset_type], y[dataset_type] = load_raw_feature_for_training(dataset[dataset_type],
                                                                         labeler,
                                                                         detector,
                                                                         predictor,
                                                                         friend)
    print("\n ---> Extract feature <--- ")
    return compute_raw_feature_for_training(X['train'], y['train'], X['val'], y['val'], friend)


def generation_datasets(images_with_friend, images_without_friend, train_ratio, friend):
    # Init parameters
    sources = [images_without_friend, images_with_friend]
    desc_train = ["Fill trainingSet set of No ", "Fill trainingSet set of "]
    desc_val = ["Fill validation set of No ", "Fill validation set of "]
    dataset = generate_empty_dataset()

    # Distribute samples between sets
    for src in sources:
        # Init parameters
        folder_path = src['path']
        files = os.listdir(folder_path)
        has_feature = src['type']
        number_of_files = int(len(files) * train_ratio)

        # Fill trainingSet
        for file_name in tqdm(random.sample(files, number_of_files),
                              desc=desc_train[has_feature] + str(friend)):
            dataset['train'][has_feature].append(TrainImage(file_name, folder_path))

        # Fill validationSet
        for file_name in tqdm(files, desc=desc_val[has_feature] + str(friend)):
            if not any(train_image.is_equal(file_name) for train_image in dataset['train'][has_feature]):
                dataset['val'][has_feature].append(TrainImage(file_name, folder_path))
    return dataset


def generation_features(images_with_friend, images_without_friend, train_ratio, predictor, detector, friend):
    print("\nDataset...")
    dataset_images = generation_datasets(images_with_friend,
                                         images_without_friend,
                                         train_ratio,
                                         friend)
    print("\nLabeler...")
    labeler = Labeler(dataset_images)
    print("\nFeature...")
    return extract_features_for_training(dataset_images,
                                         labeler,
                                         detector,
                                         predictor,
                                         friend)


def training(X_train, y_train, X_val, y_val, fitted_model_path, metrics=True):
    model = Classifier()
    print("\nFit...")
    y_pred = model.fit(X_train, y_train, X_val)
    if metrics:
        print_metrics(y_val, y_pred)
    model.save(fitted_model_path)
    return y_pred


def train_model(config, friends, detector, predictor, train_ratio=0.7):
    for friend in friends:
        # Init parameters
        images_with_friend = get_image_with_friend(config, friend)
        images_without_friend = get_image_without_friend(config, friend)
        fitted_model = get_model(config, friend)
        friend = Friend(friend)
        if not os.path.exists(fitted_model):
            os.makedirs(fitted_model)

        # Generate feature
        X_train, y_train, X_val, y_val = generation_features(images_with_friend,
                                                             images_without_friend,
                                                             train_ratio,
                                                             predictor,
                                                             detector,
                                                             friend)

        # Train model
        training(X_train,
                 y_train,
                 X_val,
                 y_val,
                 fitted_model)
