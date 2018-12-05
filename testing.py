'''
    This script compare the results for different face detection models 
    and it's result for facial landmarks
'''

from imutils import face_utils, video, resize
import dlib
import numpy as np
import cv2
import argparse


W = './shape_predictor_68_face_landmarks.dat'
P = '../face_detection/deploy.prototxt.txt'
M = '../face_detection/res10_300x300_ssd_iter_140000.caffemodel'
T = 0.6

predictor = dlib.shape_predictor(W)

image = cv2.imread("/home/keyur-r/image_data/keyur.jpeg")
# image = resize(image, height=600)
# Converting the image to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# This is based on SSD deep learning pretrained model
dl_detector = cv2.dnn.readNetFromCaffe(P, M)
hog_detector = dlib.get_frontal_face_detector()


# Facial landmarks with HOG
rects = hog_detector(gray, 0)
# For each detected face, find the landmark.
for (i, rect) in enumerate(rects):
    # Finding points for rectangle to draw on face
    x1, y1, x2, y2, w, h = rect.left(), rect.top(), rect.right() + \
        1, rect.bottom() + 1, rect.width(), rect.height()
    print(x1, y1, x2, y2)
    # Make the prediction and transfom it to numpy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    cv2.rectangle(image, (x1, y1), (x2, y2), (205, 92, 92), 2)
    # Draw on our image, all the finded cordinate points (x,y)
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (205, 92, 92), -1)

# Facial landmarks with DL
# https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7
inputBlob = cv2.dnn.blobFromImage(cv2.resize(
    image, (300, 300)), 1, (300, 300), (104, 177, 123))

dl_detector.setInput(inputBlob)
detections = dl_detector.forward()

for i in range(0, detections.shape[2]):
    # Probability of prediction
    prediction_score = detections[0, 0, i, 2]
    if prediction_score < T:
        continue
    # Finding height and width of frame
    (h, w) = image.shape[:2]
    # compute the (x, y)-coordinates of the bounding box for the
    # object
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (x1, y1, x2, y2) = box.astype("int")
    y1, x2 = int(y1 * 1.15), int(x2 * 1.05)
    print(x1, y1, x2, y2)
    # Make the prediction and transfom it to numpy array
    shape = predictor(gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
    shape = face_utils.shape_to_np(shape)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # Draw on our image, all the finded cordinate points (x,y)
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

img_height, img_width = image.shape[:2]
cv2.putText(image, "HOG", (img_width - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (205, 92, 92), 2)
cv2.putText(image, "DL", (img_width - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 255), 2)

# show the output frame
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
