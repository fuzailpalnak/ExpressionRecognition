#This file is used to create trainind data from downloaded ck+ database
import cv2, glob, random, math, numpy as np
import dlib
import itertools
from sklearn.svm import SVC
import cPickle as pickle
from sklearn.metrics import precision_score

from sklearn.metrics import confusion_matrix
from classifier import SVM as s

from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt

video_capture = cv2.VideoCapture(0)  # Webcam object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
emotions = ["anger","happy","surprise","neutral","sad","disgust"]  # Emotion list
detector = dlib.get_frontal_face_detector() # Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True,
          tol=1e-3, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
datafile="/home/palnak/facialexpression/train.pkl"
datafile1="/home/palnak/facialexpression/test.pkl"

def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("/home/palnak/facialexpression/sorted2/%s/*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction


def get_landmarks(image):
    cascade_path = "params/haarcascade_frontalface_default.xml"
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascade_path)

    # create the landmark predictor
    predictor = dlib.shape_predictor(predictor_path)


    # convert the image to grayscale

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    print "Found {0} faces!".format(len(faces))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        head = cv2.cvtColor(image[y:y + h, x:x + w],
                            cv2.COLOR_RGB2GRAY)

        # Converting the OpenCV rectangle coordinates to Dlib rectangle
        dlib_rect = dlib.rectangle(x.astype(long), y.astype(long), (x + w).astype(long), (y + h).astype(long))
        print dlib_rect

        detected_landmarks = predictor(image, dlib_rect).parts()

        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
    return np.array(landmarks),(x,y)


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale

            landmarks_vectorised = get_landmarks(gray)
            #print landmarks_vectorised.flatten()
            if landmarks_vectorised == "error":
                pass
            else:
                training_data.append(landmarks_vectorised.flatten())
                print training_data# append image array to training data list
                training_labels.append(emotion)
        print "here"
        f = open(datafile, 'wb')
        pickle.dump(training_data, f)
        pickle.dump(training_labels, f)
        f.close()

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            landmarks_vectorised = get_landmarks(gray)
            if landmarks_vectorised == "error":
                pass
            else:
                prediction_data.append(landmarks_vectorised.flatten())
                prediction_labels.append(emotion)
        f = open(datafile1, 'wb')
        pickle.dump(prediction_data, f)
        pickle.dump(prediction_labels, f)
        f.close()
    return training_data, training_labels, prediction_data, prediction_labels


accur_lin = []
for i in range(0, 1):
    a=cv2.imread("/home/palnak/Downloads/surprise1.jpg")
    landmark1,(x,y)=get_landmarks(a)
    landmark12 = np.squeeze(np.array(landmark1.flatten())).astype(np.float32)



    print("Making sets %s" % i)  # Make sets by random sampling 80/20%
    #training_data, training_labels, prediction_data, prediction_labels = make_sets()
    #print "done creating file"
    f = open("datasets/train.pkl", 'rb')
    training_data = pickle.load(f)
    training_labels = pickle.load(f)
    f.close()
    #print training_data
    f = open("datasets/test.pkl", 'rb')
    prediction_data = pickle.load(f)
    prediction_labels = pickle.load(f)
    f.close()

    X_train = np.squeeze(np.array(training_data)).astype(np.float32)
    y_train = np.array(training_labels)
    X_test = np.squeeze(np.array(prediction_data)).astype(np.float32)
    y_test = np.array(prediction_labels)
    # print np.array(y_train)


    print("training SVM linear %s" % i)  # train SVM
    #print X_train
    clf.fit(X_train, y_train)

    print("getting accuracies %s" % i)  # Use score() function to get accuracy

    pred_lin = clf.score(X_test, prediction_labels)
    Y_vote= clf.predict(X_test)
    y1=clf.predict(landmark12)
    print y1
    print clf.predict_proba(landmark12)
    print "linear: ", pred_lin
    accur_lin.append(pred_lin)  # Store accuracy in a list

    confusion = confusion_matrix(prediction_labels, Y_vote)
    print confusion
    cv2.putText(a, str(y1[0]), (x, y - 20),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("d",a)
    cv2.imwrite("fuzail.jpg",a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
print("Mean value lin svm: %.3f" % np.mean(accur_lin))  # Get mean accuracy of the 10 runs