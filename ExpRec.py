
import cv2
import numpy as np
import dlib
import time
import wx
from os import path
import cPickle as pickle

# from datasets import homebrew
from Facedetection import FaceDetector
from classifier import SVM
from gui import BaseLayout
import pca
import Landmark_Extract





class FaceLayout(BaseLayout):


    def _init_custom_layout(self):

        self.samples = []
        self.labels = []


        self.Bind(wx.EVT_CLOSE, self._on_exit)

    def init_algorithm(
            self,
            save_training_file='datasets/train.pkl',
            load_svm='params/svm.pkl',
            face_casc='params/haarcascade_frontalface_default.xml',

            landmark_data="shape_predictor_68_face_landmarks.dat",
    ):
        self.predictor = dlib.shape_predictor(landmark_data)
        self.data_file = save_training_file
        self.faces = FaceDetector(face_casc)
        # ----------Initialize the FaceDetector Constructor-----------------

        self.landmark_points = None
        self.dlib_rect=None

        # load preprocessed dataset to access labels and PCA params
        if path.isfile(save_training_file):
            (X_train, y_train) = Landmark_Extract.load_data(
                "datasets/train.pkl", test_split=0.2, seed=40)
            (X_test, y_test) = Landmark_Extract.load_test_data(
                "datasets/test.pkl", test_split=0.2, seed=40)

            self.all_labels = np.unique(np.hstack(y_train))
            self. num_classes =len(self.all_labels)

            # load pre-trained multi-layer perceptron
            if path.isfile(load_svm):
                # if path.isfile(load_mlp):
                # layer_sizes = np.array([self.pca_V.shape[1],len(self.all_labels)])
                self.SVM = SVM(self.all_labels ,self.num_classes)
                self.SVM.load(load_svm)
                print self.SVM
                # svm=cv2.SVM()
                # svm.load(load_svm)

            else:
                print "Warning: Testing is disabled"
                print "Could not find pre-trained MLP file ", load_svm
                self.testing.Disable()
        else:
            print "Warning: Testing is disabled"
            print "Could not find data file ", save_training_file
            self.testing.Disable()

    def _create_custom_layout(self):

        pnl1 = wx.Panel(self, -1)
        self.training = wx.RadioButton(pnl1, -1, 'Train', (10, 10),
                                       style=wx.RB_GROUP)
        self.Bind(wx.EVT_RADIOBUTTON, self._on_training, self.training)
        self.testing = wx.RadioButton(pnl1, -1, 'Test')
        self.Bind(wx.EVT_RADIOBUTTON, self._on_testing, self.testing)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(self.training, 1)
        hbox1.Add(self.testing, 1)
        pnl1.SetSizer(hbox1)


        pnl2 = wx.Panel(self, -1)
        self.angry = wx.RadioButton(pnl2, -1, 'angry', (10, 10),
                                      style=wx.RB_GROUP)
        self.happy = wx.RadioButton(pnl2, -1, 'happy')
        self.surprise = wx.RadioButton(pnl2, -1, 'surprise')
        self.neutral = wx.RadioButton(pnl2, -1, 'neutral')
        self.sad = wx.RadioButton(pnl2, -1, 'sad')
        self.disgust= wx.RadioButton(pnl2, -1, 'disgust')
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(self.angry, 1)
        hbox2.Add(self.happy, 1)
        hbox2.Add(self.surprise, 1)
        hbox2.Add(self.neutral, 1)
        hbox2.Add(self.sad, 1)
        hbox2.Add(self.disgust, 1)
        pnl2.SetSizer(hbox2)

        # create horizontal layout with single snapshot button
        pnl3 = wx.Panel(self, -1)
        self.snapshot = wx.Button(pnl3, -1, 'Take Snapshot')
        self.Bind(wx.EVT_BUTTON, self._on_snapshot, self.snapshot)
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3.Add(self.snapshot, 1)
        pnl3.SetSizer(hbox3)

        # arrange all horizontal layouts vertically
        self.panels_vertical.Add(pnl1, flag=wx.EXPAND | wx.TOP, border=1)
        self.panels_vertical.Add(pnl2, flag=wx.EXPAND | wx.BOTTOM, border=1)
        self.panels_vertical.Add(pnl3, flag=wx.EXPAND | wx.BOTTOM, border=1)

    def _process_frame(self, frame):

        # -------------------------- detect face----------------------------------------
        success, self.frame, self.landmark_points, self.dlib_rect,(x,y) = self.faces.detect(frame)


        # since testing is disabled 'if' part won't run until dataset is trained atleast once

        if success and self.testing.GetValue():


            success, landmark_points = self.faces.align_head(self.landmark_points,self.dlib_rect,self.predictor,self.frame)

            if success:




                # predict label with pre-trained MLP
                label = self.SVM.predict(landmark_points.flatten())

                # draw label above bounding box
                cv2.putText(frame, str(label), (x, y - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


        return frame

    def _on_training(self, evt):
        """Enables all training-related buttons when Training Mode is on"""
        self.neutral.Enable()
        self.happy.Enable()
        self.sad.Enable()
        self.surprise.Enable()
        self.angry.Enable()
        self.disgust.Enable()
        self.snapshot.Enable()

    def _on_testing(self, evt):
        """Disables all training-related buttons when Testing Mode is on"""
        self.neutral.Disable()
        self.happy.Disable()
        self.sad.Disable()
        self.surprise.Disable()
        self.angry.Disable()
        self.disgust.Disable()
        self.snapshot.Disable()

    def _on_snapshot(self, evt):
        """Takes a snapshot of the current frame

            This method takes a snapshot of the current frame, preprocesses
            it to extract the head region, and upon success adds the data
            sample to the training set.
        """
        if self.neutral.GetValue():
            label = 'anger'
        elif self.happy.GetValue():
            label = 'happy'
        elif self.sad.GetValue():
            label = 'surprise'
        elif self.surprised.GetValue():
            label = 'neutral'
        elif self.angry.GetValue():
            label = 'sad'
        elif self.disgusted.GetValue():
            label = 'disgust'

        if self.landmark_points is None:
            print "No face detected"
        else:

            success, landmark_points = self.faces.align_head(self.landmark_points,self.dlib_rect,self.predictor,self.frame)

            #print landmark_points.flatten()
            if success:
                print "Added sample to training set"
                self.samples.append(landmark_points.flatten())
                print self.samples
                self.labels.append(label)
            else:
                print "Could not align head (eye detection failed?)"

    def _on_exit(self, evt):
        """Dumps the training data to file upon exiting"""
        # if we have collected some samples, dump them to file
        if len(self.samples) > 0:
            # make sure we don't overwrite an existing file
            if path.isfile(self.data_file):
                # file already exists, construct new load_from_file
                load_from_file, fileext = path.splitext(self.data_file)
                offset = 0
                while True:
                    file = load_from_file + "-" + str(offset) + fileext
                    if path.isfile(file):
                        offset += 1
                    else:
                        break
                self.data_file = file

            # dump samples and labels to file
            f = open(self.data_file, 'wb')
            pickle.dump(self.samples, f)
            pickle.dump(self.labels, f)
            f.close()

            # inform user that file was created
            print "Saved", len(self.samples), "samples to", self.data_file

        # deallocate
        self.Destroy()


def main():
    capture = cv2.VideoCapture(0)
    if not(capture.isOpened()):
        capture.open()

    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # start graphical user interface
    app = wx.App()
    layout = FaceLayout(capture, title='Facial Expression Recognition')
    layout.init_algorithm()
    layout.Show(True)
    app.MainLoop()


if __name__ == '__main__':
    main()
