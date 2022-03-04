faceProto = "../libs/opencv_face_detector.pbtxt"
faceModel = "../libs/opencv_face_detector_uint8.pb"
faceNet = cv.dnn.readNet(faceModel, faceProto)
predictor = dlib.shape_predictor('../libs/shape_predictor_68_face_landmarks.dat')
face_cascade = cv.CascadeClassifier('../libs/haarcascade_frontalface_default.xml')
faceNet = cv.dnn.readNet(faceModel, faceProto)
path_source = "./datasetDetectedFaces/source"
destination_images = "./datasetDetectedFaces/extracted_faces"
