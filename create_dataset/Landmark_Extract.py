#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy as np



from os import path
import cPickle as pickle


def load_data(load_from_file, test_split=0.2,  plot_samples=False, seed=40):


    X = []
    labels = []
    if not path.isfile(load_from_file):
        print "Could not find file", load_from_file
        return (X, labels), (X, labels), None, None
    else:
        print "Loading data from", load_from_file
        f = open(load_from_file, 'rb')
        samples = pickle.load(f)
        labels = pickle.load(f)
        f.close()
        print "Loaded", len(samples), "training samples"



        if plot_samples:
            print "Plotting samples not implemented"

    # shuffle dataset
    # np.random.seed(seed)
    # np.random.shuffle(samples)
    # np.random.seed(seed)
    # np.random.shuffle(labels)

    # split data according to test_split




    return samples,labels




def load_test_data(load_from_file, test_split=0.2,  plot_samples=False, seed=40):


    X = []
    labels = []
    if not path.isfile(load_from_file):
        print "Could not find file", load_from_file
        return (X, labels), (X, labels), None, None
    else:
        print "Loading data from", load_from_file
        f = open(load_from_file, 'rb')
        samples = pickle.load(f)
        labels = pickle.load(f)
        f.close()
        print "Loaded", len(samples), "training samples"



        if plot_samples:
            print "Plotting samples not implemented"

    # shuffle dataset
    np.random.seed(seed)
    np.random.shuffle(samples)
    np.random.seed(seed)
    np.random.shuffle(labels)

    # split data according to test_split




    return samples,labels