#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module that contains various classifiers"""

import cv2
import numpy as np

from abc import ABCMeta, abstractmethod
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import datasets
import cPickle as pickle
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score
#from sklearn.model_selection import cross_val_score




class Classifier:

    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test, visualize=False):
        pass



    def _precision(self, y_test, Y_vote,mode="one-vs-one"):

        # predicted classes
        #y_hat = np.argmax(Y_vote, axis=1)
        self.mode=mode

        if self.mode == "one-vs-one":
            # need confusion matrix
            conf  = confusion_matrix(y_test, Y_vote)

            # consider each class separately
            prec = np.zeros(self.num_classes)
            for c in xrange(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = conf[c, c]

                # false positives: label is c, classifier predicted not c
                fp = np.sum(conf[:, c]) - conf[c, c]

                # precision
                if tp + fp != 0:
                    prec[c] = tp * 1. / (tp + fp)
        elif self.mode == "one-vs-all":
            # consider each class separately
            prec = np.zeros(self.num_classes)
            for c in xrange(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = np.count_nonzero((y_test == c) * (Y_vote == c))

                # false positives: label is c, classifier predicted not c
                fp = np.count_nonzero((y_test == c) * (Y_vote != c))

                if tp + fp != 0:
                    prec[c] = tp * 1. / (tp + fp)

        return prec

    def _recall(self, y_test, Y_vote,mode="one-vs-one"):

        # predicted classes
        y_hat = np.argmax(Y_vote, axis=-1)
        self.mode=mode
        if self.mode == "one-vs-one":
            # need confusion matrix
            conf =  confusion_matrix(y_test, Y_vote)

            # consider each class separately
            recall = np.zeros(self.num_classes)
            for c in xrange(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = conf[c, c]

                # false negatives: label is not c, classifier predicted c
                fn = np.sum(conf[c, :]) - conf[c, c]
                if tp + fn != 0:
                    recall[c] = tp * 1. / (tp + fn)
        elif self.mode == "one-vs-all":
            # consider each class separately
            recall = np.zeros(self.num_classes)
            for c in xrange(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = np.count_nonzero((y_test == c) * (y_hat == c))

                # false negatives: label is not c, classifier predicted c
                fn = np.count_nonzero((y_test != c) * (y_hat == c))

                if tp + fn != 0:
                    recall[c] = tp * 1. / (tp + fn)
        return recall



class SVM(Classifier):


    def __init__(self,class_labels,num_classes
                 ):

        self.num_classes=num_classes
        self.class_labels = class_labels


    def load(self, file,face_model=False):

        f = open(file, 'rb')
        self.model = pickle.load(f)
        print self.model

        f.close()
        if face_model:
            return self.model



    def fit(self, X_train, y_train):

        print "Starting to train Model "
        y_train = self._labels_str_to_num(y_train)

        # train model

        self.model = svm.SVC(kernel='linear', probability=True, tol=1e-3, verbose = True)
        self.model.fit(X_train, y_train)
        #----save the svm model--------------
        f = open("params/svm_normalized.pkl", 'wb')
        pickle.dump(self.model,f)
        print "Creating Model Complete-------"

    def predict(self, X_test):
        X_test = np.asarray(X_test)
        y_hat= self.model.predict(X_test)


        # find the most active cell in the output layer
        #y_hat = np.argmax(y_hat, 1)

        # return y_hat
        return self.__labels_num_to_str(y_hat),self.model.predict_proba(X_test)

    def evaluate(self, X_test, y_test):

        # need int labels
        y_test = self._labels_str_to_num(y_test)
        pred_lin = self.model.score(X_test, y_test)
        print "accuracy"
        print  pred_lin
        # predict labels
        Y_vote = self.model.predict(X_test)

        confusion = confusion_matrix(y_test, Y_vote)
        print confusion
        print "Precision score-"+ str(precision_score(y_test,Y_vote))
        print "Recall score-"+str(recall_score(y_test,Y_vote))




    def _one_hot(self, y_train):
        """Converts a list of labels into a 1-hot code"""
        numSamples = len(y_train)
        new_responses = np.zeros(numSamples*self.num_classes, np.float32)
        resp_idx = np.int32(y_train + np.arange(numSamples)*self.num_classes)
        new_responses[resp_idx] = 1
        return new_responses

    def _labels_str_to_num(self, labels):
        """Converts a list of string labels to their corresponding ints"""
        return np.array([int(np.where(self.class_labels == l)[0])
                         for l in labels])

    def __labels_num_to_str(self, labels):
        """Converts a list of int labels to their corresponding strings"""
        return self.class_labels[labels]


