import cv2 as cv
import dlib
import numpy as np


def is_colored(image):
    return image.shape == 3


def denoise_preserving_edge_from(image, epsilon=0.01, neighborhood_pixels=3):
    if is_colored(image):
        cv.ximgproc.guidedFilter(image, image, neighborhood_pixels, epsilon, image)
    else:
        cv.ximgproc.guidedFilter(image, image, neighborhood_pixels, epsilon, image, -1)
    return image


def remove_color_from(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def remove_wrinkles_from(image):
    return cv.bilateralFilter(image, 9, 200, 200)


def isolate_specific_edges_from(image, orientations, kernel_gabor=(3, 3), standard_deviation=45, wavelength=3.1,
                                spatial_ratio=0.25, phase_offset=0):
    kernel = cv.getGaborKernel(kernel_gabor, standard_deviation, orientations[0], wavelength, spatial_ratio,
                               phase_offset,
                               ktype=cv.CV_32F)
    filtered_image = cv.filter2D(image, cv.CV_8UC3, kernel)
    for orientation in orientations[1:]:
        kernel = cv.getGaborKernel(kernel_gabor, standard_deviation, orientation, wavelength, spatial_ratio,
                                   phase_offset,
                                   ktype=cv.CV_32F)
        filtered_image += cv.filter2D(image, cv.CV_8UC3, kernel)
    return filtered_image


def threshold_from(image):
    return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


def remove_shadow_from(image):
    rgb_channels = cv.split(image)
    results = []
    for image_chanel in rgb_channels:
        kernel_dilation = np.ones((3, 3), np.uint8)
        dilated_image = cv.dilate(image_chanel, kernel_dilation)
        kernel_median = 9
        median_image = cv.medianBlur(dilated_image, kernel_median)
        difference_image = 255 - cv.absdiff(image_chanel, median_image)
        results.append(
            cv.normalize(difference_image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1))
    return cv.merge(results)


def boosted_high_pass_filter_from(image, sigma=7, boost_sharpening=3):
    smoothing = cv.GaussianBlur(image, (sigma, sigma), 0, 0, cv.BORDER_REPLICATE)
    edges = image - smoothing
    return image + boost_sharpening * edges


def concatenate_black_pixels_from(image, size_kernel_opening=3):
    kernel_opening = np.ones((size_kernel_opening, size_kernel_opening), np.uint8)
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel_opening)


def improve_edge_from(image):
    sharpened_image = boosted_high_pass_filter_from(image)
    return concatenate_black_pixels_from(sharpened_image)


def clean_upper_face_from(image):
    image = remove_color_from(image)
    # image = boosted_high_pass_filter_from(image)
    angle_edge_to_detect = [90]
    image = isolate_specific_edges_from(image, angle_edge_to_detect)
    # image = threshold_from(image)
    # image = remove_wrinkles_from(image)
    return image


def crop_by_landmarks(image, landmarks, range_face, landmark_y_min, landmark_y_max):
    under_face_bridge_x = []
    under_face_bridge_y = []
    for i in range_face:
        under_face_bridge_x.append(landmarks[i][0])
        under_face_bridge_y.append(landmarks[i][1])
    x_min = min(under_face_bridge_x)
    x_max = max(under_face_bridge_x)
    y_min = landmarks[landmark_y_min][1]
    y_max = landmarks[landmark_y_max][1]
    return image[y_min:y_max, x_min:x_max]


def preprocessing(image, detector, predictor):
    try:
        image = denoise_preserving_edge_from(image)

        rect = detector(image)[0]
        sp = predictor(image, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])

        range_face = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        landmark_y_min = 19
        landmark_y_max = 8
        size = 100
        face = crop_by_landmarks(image, landmarks, range_face, landmark_y_min, landmark_y_max)
        face = cv.resize(face, (size, size), cv.INTER_LANCZOS4)

        return clean_upper_face_from(face)
    except:
        return None


if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../libs/shape_predictor_68_face_landmarks.dat')
    image = preprocessing(cv.imread("../data/trainingSet/Angelo/27.jpg"), detector, predictor)
    cv.imshow("proof", image)
    cv.waitKey(0)
