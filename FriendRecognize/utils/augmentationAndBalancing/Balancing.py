import os
import random

from FriendRecognize.utils.UsefulMethods import load_images_from


def removing(path, remaining):
    images = load_images_from(path, get_images=False)
    while remaining > 0:
        image_name = random.choice(images)
        os.remove(path + "/" + image_name)
        images.remove(image_name)
        remaining -= 1


def balance(paths):
    for index in range(len(paths)):
        if index % 2 == 0:
            number_elements = len([name for name in os.listdir(paths[index])])
            number_no_elements = len([name for name in os.listdir(paths[index + 1])])
            remaining = max(number_elements, number_no_elements) - min(number_elements, number_no_elements)
            if number_elements > number_no_elements:
                removing(paths[index], remaining)
            else:
                removing(paths[index + 1], remaining)
