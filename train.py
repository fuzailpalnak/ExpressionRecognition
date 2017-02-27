#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A script to train and test classifier"""

import numpy as np
from classifier import SVM
import Landmark_Extract



def main():
    (X_train, y_train)= Landmark_Extract.load_data(
        "datasets/train.pkl",test_split=0.2,seed=40)
    (X_test, y_test) = Landmark_Extract.load_test_data(
        "datasets/test.pkl", test_split=0.2, seed=40)






    X_train = np.squeeze(np.array(X_train)).astype(np.float32)
    y_train = np.array(y_train)
    X_test = np.squeeze(np.array(X_test)).astype(np.float32)
    y_test = np.array(y_test)


    labels = np.unique(np.hstack((y_train)))

    num_features = len(X_train[0])

    num_classes = len(labels)
    Svm = SVM(labels,num_classes)
    Svm.fit(X_train,y_train)
    Svm.evaluate(X_test,y_test)


if __name__ == '__main__':
    main()
