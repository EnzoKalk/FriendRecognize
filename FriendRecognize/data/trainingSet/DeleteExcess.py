import os
import random
import sys

import cv2 as cv
from tqdm import tqdm


def removing(path, remaining):
    images = load_images_from_folder(path)
    while remaining > 0:
        img = random.choice(images)
        os.remove(path + "/" + img[0])
        images.remove(img)
        remaining -= 1


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        images.append((filename, img))
    return images


if __name__ == "__main__":
    # Paths
    path_vincenzo = './Vincenzo'
    path_angelo = './Angelo'
    path_dima = './Dima'
    path_giovanna = './Giovanna'
    path_noemi = './Noemi'

    path_noVincenzo = './noVincenzo'
    path_noAngelo = './noAngelo'
    path_noDima = './noDima'
    path_noGiovanna = './noGiovanna'
    path_noNoemi = './noNoemi'

    number_vincenzo_elements = len([name for name in tqdm(os.listdir(path_vincenzo), desc="Count Vincenzo elements")])
    number_angelo_elements = len([name for name in tqdm(os.listdir(path_angelo), desc="Count Angelo elements")])
    number_dima_elements = len([name for name in tqdm(os.listdir(path_dima), desc="Count Dima elements")])
    number_giovanna_elements = len([name for name in tqdm(os.listdir(path_giovanna), desc="Count Giovanna elements")])
    number_noemi_elements = len([name for name in tqdm(os.listdir(path_noemi), desc="Count Noemi elements")])

    number_noVincenzo_elements = len(
        [name for name in tqdm(os.listdir(path_noVincenzo), desc="Count Vincenzo elements")])
    number_noAngelo_elements = len([name for name in tqdm(os.listdir(path_noAngelo), desc="Count Angelo elements")])
    number_noDima_elements = len([name for name in tqdm(os.listdir(path_noDima), desc="Count Dima elements")])
    number_noGiovanna_elements = len(
        [name for name in tqdm(os.listdir(path_noGiovanna), desc="Count Giovanna elements")])
    number_noNoemi_elements = len([name for name in tqdm(os.listdir(path_noNoemi), desc="Count Noemi elements")])

    if number_vincenzo_elements > number_noVincenzo_elements:
        removing(path_vincenzo,
                 max(number_vincenzo_elements, number_noVincenzo_elements) - min(number_vincenzo_elements,
                                                                                 number_noVincenzo_elements))
    else:
        removing(path_noVincenzo,
                 max(number_vincenzo_elements, number_noVincenzo_elements) - min(number_vincenzo_elements,
                                                                                 number_noVincenzo_elements))

    if number_angelo_elements > number_noAngelo_elements:
        removing(path_angelo, max(number_angelo_elements, number_noAngelo_elements) - min(number_angelo_elements,
                                                                                          number_noAngelo_elements))
    else:
        removing(path_noAngelo, max(number_angelo_elements, number_noAngelo_elements) - min(number_angelo_elements,
                                                                                            number_noAngelo_elements))

    if number_dima_elements > number_noDima_elements:
        removing(path_dima,
                 max(number_dima_elements, number_noDima_elements) - min(number_dima_elements, number_noDima_elements))
    else:
        removing(path_noDima,
                 max(number_dima_elements, number_noDima_elements) - min(number_dima_elements, number_noDima_elements))

    if number_giovanna_elements > number_noGiovanna_elements:
        removing(path_giovanna,
                 max(number_giovanna_elements, number_noGiovanna_elements) - min(number_giovanna_elements,
                                                                                 number_noGiovanna_elements))
    else:
        removing(path_noGiovanna,
                 max(number_giovanna_elements, number_noGiovanna_elements) - min(number_giovanna_elements,
                                                                                 number_noGiovanna_elements))

    if number_noemi_elements > number_noNoemi_elements:
        removing(path_noemi, max(number_noemi_elements, number_noNoemi_elements) - min(number_noemi_elements,
                                                                                       number_noNoemi_elements))
    else:
        removing(path_noNoemi, max(number_noemi_elements, number_noNoemi_elements) - min(number_noemi_elements,
                                                                                         number_noNoemi_elements))

    # Exit
    sys.exit(0)
