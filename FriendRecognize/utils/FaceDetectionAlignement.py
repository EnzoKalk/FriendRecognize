import os

import dlib

from FriendRecognize.utils.FaceAligner import FaceAligner

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2


def pad_bb(rect, shape, padding=20):
    # Add padding to the bbox taking into account the image shape
    rect[0] = max(0, rect[0] - padding)
    rect[1] = max(0, rect[1] - padding)
    rect[2] = min(rect[2] + padding, shape[1] - 1)
    rect[3] = min(rect[3] + padding, shape[0] - 1)

    return rect


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencv2Dnn = frame.copy()
    frameHeight = frameOpencv2Dnn.shape[0]
    frameWidth = frameOpencv2Dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencv2Dnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
    return frameOpencv2Dnn, bboxes


def alignement(img, faceNet, predictor):
    fa = FaceAligner(predictor, desiredFaceWidth=224, desiredFaceHeight=224)
    # Find box of the image
    frameFace, bboxes = getFaceBox(faceNet, img)
    padding = 0
    showface = frameFace.copy()
    faceim = None
    if frameFace is not None:
        for bbox in bboxes:
            bbox = pad_bb(bbox, frameFace.shape, padding)
            dlibRect = dlib.rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            grayframe = cv2.cvtColor(frameFace, cv2.COLOR_BGR2GRAY)
            faceim = fa.align(frameFace, grayframe, dlibRect)
            try:
                cv2.resize(frameFace[bbox[1]:bbox[3], bbox[0]:bbox[2]], (224, 224), cv2.INTER_LANCZOS4)
            except Exception as e:
                return None
            x1, y1, x2, y2 = bbox
            c = (0, 255, 0)
            cv2.rectangle(showface, (x1, y1), (x2, y2), c, 3, 8)
        return faceim
    return None
