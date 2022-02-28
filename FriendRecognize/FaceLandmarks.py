# Import required modules
import sys, os, math, time, argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Dlib demo. Esc to exit.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
args = parser.parse_args()

# Load detector
import dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor( 'shape_predictor_68_face_landmarks.dat' )


# Open a video file or an image file or a camera stream
cap = cv2.VideoCapture(args.input if args.input else 0)
framen = 0
while True:
    # Read frame
    t = time.time()
    hasFrame, frame = cap.read()
    framen+=1
    if framen %10:
        continue

    if not hasFrame:
        cv2.waitKey()
        break
    
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(grayframe, 1)


    showface = frame.copy()
    for rect in faces:
        bbox = [ rect.left(), rect.top(), rect.right(), rect.bottom()]
        shape = predictor(grayframe, rect)

        x1, y1, x2, y2 = bbox
        c =  (0, 255, 0)
        for i in range(0,68):
            p = (shape.part(i).x, shape.part(i).y)
            cv2.circle(showface, p, 2, c, -1)
    cv2.imshow("facial landmarks", showface)
    k = cv2.waitKey(40) & 0xff
    if k==27:
        break
