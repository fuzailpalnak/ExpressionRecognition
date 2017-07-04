#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A script to train and test classifier"""

import numpy as np
from classifier.classifier import SVM
from create_dataset import createDataset
import os
from create_dataset import Landmark_Extract



def main(train_pickle,test_pickle):
    if os.path.exists(train_pickle) and os.path.exists(test_pickle):
        (X_train, y_train)= Landmark_Extract.load_data(
            "datasets/train.pkl",test_split=0.2,seed=40)
        (X_test, y_test) = Landmark_Extract.load_test_data(
            "datasets/test.pkl", test_split=0.2, seed=40)
    else:
        X_train, y_train, X_test, y_test=createDataset.create_dataset()







    X_train =np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)


    labels = np.unique(np.hstack((y_train)))

    num_features = len(X_train[0])

    num_classes = len(labels)
    Svm = SVM(labels,num_classes)
    Svm.fit(X_train,y_train)
    Svm.evaluate(X_test,y_test)


if __name__ == '__main__':
    train_pickle = "/home/palnak/PycharmProjects/ExpRec/datasets/train.pkl"
    test_pickle = "/home/palnak/PycharmProjects/ExpRec/datasets/test.pkl"
    main(train_pickle,test_pickle)