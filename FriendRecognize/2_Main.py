import dlib
import yaml

from FriendRecognize.utils.augmentationAndBalancing.Augmentation import augment
from FriendRecognize.utils.augmentationAndBalancing.Balancing import balance
from FriendRecognize.utils.trainModelAndClassifier.TrainModel import train


def get_friends(config):
    return config['friends']['with']


def get_with_feature(config, feature):
    return config['data']['training'] + config['friends']['with'][feature]


def get_without_feature(config, feature):
    return config['data']['training'] + config['friends']['without'][feature]


def get_predictor_path(config):
    return config['libs']['predictor']


if __name__ == '__main__':
    # Init parameters
    with open('config.yml') as file:
        config = yaml.full_load(file)
    friends = get_friends(config)
    perform_augmentation = False
    perform_balancing = False
    perform_training = False

    # Check answers
    answer = ""
    print("Do you want perform augmentation?[y/n]")
    while not (answer == "y" or answer == "n"):
        answer = input().lower()
        if not (answer == "y" or answer == "n"):
            print("TRY AGAIN!! Do you want perform augmentation?[y/n]")
    if answer == "y":
        perform_augmentation = True

    answer = ""
    print("Do you want perform balancing?[y/n]")
    while not (answer == "y" or answer == "n"):
        answer = input().lower()
        if not (answer == "y" or answer == "n"):
            print("TRY AGAIN!! Do you want perform balancing?[y/n]")
    if answer == "y":
        perform_balancing = True

    answer = ""
    print("Do you want perform training?[y/n]")
    while not (answer == "y" or answer == "n"):
        answer = input().lower()
        if not (answer == "y" or answer == "n"):
            print("TRY AGAIN!! Do you want perform traing?[y/n]")
    if answer == "y":
        perform_training = True

    # Perform answers
    if perform_augmentation:
        print("\nAugmentation...")
        paths = []
        for friend in friends:
            paths.append(get_with_feature(config, friend))
            paths.append(get_without_feature(config, friend))
        augment(paths)

    if perform_balancing:
        print("\nBalancing...")
        paths = []
        for friend in friends:
            paths.append(get_with_feature(config, friend))
            paths.append(get_without_feature(config, friend))
        balance(paths)

    if perform_training:
        print("\nTraining...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(get_predictor_path(config))
        train(config, friends, detector, predictor)

    exit(0)
