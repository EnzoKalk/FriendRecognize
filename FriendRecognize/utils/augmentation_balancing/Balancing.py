import os
import random

from FriendRecognize.utils.UsefulMethods import load_images_from


def removing(path, remaining):
    # Init parameters
    images = load_images_from(path, get_images=False)

    # Balancing
    while remaining > 0:
        # Init parameters
        image_name = random.choice(images)
        remaining -= 1

        # Removing
        os.remove(os.path.join(path, image_name))
        images.remove(image_name)


def balancing(paths):
    for index in range(len(paths)):
        if index % 2 == 0:
            # Init parameters
            number_elements = len([name for name in os.listdir(paths[index])])
            number_no_elements = len([name for name in os.listdir(paths[index + 1])])
            remaining = max(number_elements, number_no_elements) - min(number_elements, number_no_elements)

            # Choice path for removing
            if number_elements > number_no_elements:
                removing(paths[index], remaining)
            else:
                removing(paths[index + 1], remaining)
