"""

    Created on Mon Dec 03 11:15:45 2018
    @author: keyur-r
    
    python facial_landmarks_realtime.py -l <> -w <> -p <> -m <> -t <>

    l -> hog or dl
    w -> model path for facial landmarks (shape_predictor_68_face_landmarks.dat)
    p -> Caffe prototype file for dnn module (deploy.prototxt.txt)
    m -> Caffe trained model weights path (res10_300x300_ssd_iter_140000.caffemodel)
    t -> Thresold to filter weak face in dnn
    
"""

import numpy as np
import dlib
import cv2
import argparse
from image_utility import save_image, generate_random_color, draw_border
from imutils import face_utils, video


def hog_detector(image, gray):
    # Finding height and width of frame
    (img_h, img_w) = image.shape[:2]
    cv2.putText(image, "HOG method", (img_w - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (205, 92, 92), 2)
    # Get faces into webcam's image
    rects = detector(gray, 0)
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Finding points for rectangle to draw on face
        x1, y1, x2, y2, w, h = rect.left(), rect.top(), rect.right() + \
            1, rect.bottom() + 1, rect.width(), rect.height()

        # cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (205, 92, 92), 2)
        draw_border(image, (x1, y1), (x2, y2), (205, 92, 92), 2, 10, 20)
        # show the face number
        cv2.putText(image, "Found #{}".format(i + 1), (x1 - 20, y1 - 20),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.6, (205, 92, 92), 2)
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Draw on our image, all the finded cordinate points (x,y)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def dl_detector(image, gray):
    # Facial landmarks with DL

    # Finding height and width of frame
    (h, w) = image.shape[:2]
    cv2.putText(image, "DL method", (w - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (205, 92, 92), 2)
    total_faces = 0
    # https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7
    inputBlob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 1, (300, 300), (104, 177, 123))

    detector.setInput(inputBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):

        # Probability of prediction
        prediction_score = detections[0, 0, i, 2]
        if prediction_score < T:
            continue
        total_faces += 1

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        # For better landmark detection
        # y1, x2 = int(y1 * 1.15), int(x2 * 1.05)

        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
        shape = face_utils.shape_to_np(shape)
        # cv2.rectangle(image, (x1, y1), (x2, y2), (205, 92, 92), 2)
        draw_border(image, (x1, y1), (x2, y2), (205, 92, 92), 2, 10, 20)

        # show the face number with confidence score
        prediction_score_str = "{:.2f}%".format(prediction_score * 100)
        label = "Found #{} ({})".format(total_faces, prediction_score_str)
        cv2.putText(image, label, (x1 - 20, y1 - 20),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.6, (205, 92, 92), 2)

        # Draw on our image, all the finded cordinate points (x,y)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)


def facial_landmarks():

    # Feed from computer camera with threading
    cap = video.VideoStream(src=0).start()

    while True:
        # Getting out image by webcam
        image = cap.read()

        # Converting the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if face_detection_method == 'hog':
            hog_detector(image, gray)
        else:
            dl_detector(image, gray)

        # show the output frame
        # adding brightness and contrast -> α⋅p(i,j)+β where p(i.j) is pixel value for each point
        image = cv2.convertScaleAbs(image, alpha=1.0, beta=0)
        cv2.imshow("Facial Landmarks", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.stop()

if __name__ == "__main__":

    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--weights',
                    default='./shape_predictor_68_face_landmarks.dat', help='Path to weights file')
    ap.add_argument("-p", "--prototxt", default="../face_detection/deploy.prototxt.txt",
                    help="Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", default="../face_detection/res10_300x300_ssd_iter_140000.caffemodel",
                    help="Pre-trained caffe model")
    ap.add_argument("-t", "--thresold", type=float, default=0.6,
                    help="Thresold value to filter weak detections")
    ap.add_argument("-l", "--learning", default="hog",
                    help="Learning model from hog/dl")
    args = ap.parse_args()

    predictor = dlib.shape_predictor(args.weights)
    face_detection_method = args.learning
    T = args.thresold
    if face_detection_method == 'hog':
        detector = dlib.get_frontal_face_detector()
    elif face_detection_method == 'dl':
        detector = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)
    else:
        print("Please select method from dl or hog to find landmarks")

    facial_landmarks()
