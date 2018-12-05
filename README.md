Facial Landmarks Detection
==================
Steps to find 68 points facial landmarks using OpenCV :

1. First read image frame from disk
2. Create face detector object (please read below notes for more details)
3. Find faces from frame using face detector object
4. Find facial landmarks on faces using 'shape_predictor_68_face_landmarks.dat'

Here I have tried below face detectors for finding landmarks:

1. Dlib's get_frontal_face : based on HOG + SVM classifier
2. Dlib's cnn_face_detection_model_v1 : CNN architecture trained model mmod_human_face_detector.dat
3. OpenCV's DNN module : Pre-trained deep learning caffe model with SSD method 


Installation
==================

To start with project just follow the few steps 

    $ git clone https://github.com/keyurr2/face-detection.git
    $ pip install -r requirements.txt
    $ cd into <project-folder>

Now you need to download models from below url and put in project directory

1. http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
2. https://github.com/keyurr2/face-detection/blob/master/deploy.prototxt.txt
3. https://github.com/keyurr2/face-detection/blob/master/res10_300x300_ssd_iter_140000.caffemodel
4. https://github.com/keyurr2/face-detection/blob/master/mmod_human_face_detector.dat

To find facial landmarks in realtime with different methods

    $ python facial_landmarks_realtime.py -l hog
    $ python facial_landmarks_realtime.py -l cnn
    $ python facial_landmarks_realtime.py -l dl

To find for the same in image just give image path like 

    $ python facial_landmarks.py -l hog -i <image-path>
    
(Please change HOME in script)

### Screenshot
![HOG+SVM vs CNN vs DNN](/out.png?raw=true "HOG+SVM vs CNN vs DNN")


Authors
==================

* **Keyur Rathod (keyur.rathod1993@gmail.com)**

License
==================

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
