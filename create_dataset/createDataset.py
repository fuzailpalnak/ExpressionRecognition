#This file is used to create trainind data from downloaded ck+ database
import cv2, glob, random, math, numpy as np
import dlib
import os
import itertools
from sklearn.svm import SVC
import cPickle as pickle
from detection_recognition.Facedetection import FaceDetector

from sklearn.metrics import precision_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt






def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("/home/palnak/PycharmProjects/ExpRec/sorted2/%s/*.png" % str(emotion))
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction



def make_sets(known_emotion,faces,predictor):
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    train_pickle = "/home/palnak/PycharmProjects/ExpRec/datasets/train.pkl"
    test_pickle = "/home/palnak/PycharmProjects/ExpRec/datasets/test.pkl"
    for emotion in known_emotion:
        training, prediction = get_files(emotion)
        print "Creating dataset for - "+str(emotion)
        for item in training:
            image = cv2.imread(item)  # open image
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image=clahe.apply(image)
            image, dlib_face = faces.detect_dlib(image)
            if len(dlib_face) < 0:
                print "No face Detected"
                continue
            success, landmarks_vectorised = faces.align_head(image, dlib_face,predictor)
            if not success:
                print "Error while getting landmarks"
                continue
            else:
                if len(landmarks_vectorised.tolist())==268 or len(landmarks_vectorised.tolist())==68:
                    training_data.append(landmarks_vectorised.flatten())
                    training_labels.append(emotion)

        f = open(train_pickle, 'wb')
        pickle.dump(training_data, f)
        pickle.dump(training_labels, f)
        f.close()

        for item in prediction:
            image = cv2.imread(item)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image=clahe.apply(image)



            image, dlib_face = faces.detect_dlib(image)
            if len(dlib_face) < 0:
                print "No face Detected"
                continue
            success, landmarks_vectorised = faces.align_head(image, dlib_face,predictor)
            if not success:
                print "Error while getting landmarks"
                continue
            else:
                if len(landmarks_vectorised.tolist()) == 268 or len(landmarks_vectorised.tolist()) == 68:
                    prediction_data.append(landmarks_vectorised.flatten())
                    prediction_labels.append(emotion)
        f = open(test_pickle, 'wb')
        pickle.dump(prediction_data, f)
        pickle.dump(prediction_labels, f)
        f.close()


    return training_data, training_labels, prediction_data, prediction_labels



def create_dataset():
    landmark_data = "shape_predictor_68_face_landmarks.dat"



    predictor = dlib.shape_predictor(landmark_data)
    detector = dlib.get_frontal_face_detector()

    faces = FaceDetector( detector, predictor)

    for root, known_emotion, files in os.walk("/home/palnak/PycharmProjects/ExpRec/sorted2/", True):
        break

    return make_sets(known_emotion,faces,predictor)



if __name__ == '__main__':
    create_dataset()