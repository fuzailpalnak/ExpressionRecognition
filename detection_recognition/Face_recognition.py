import dlib
import cv2
import numpy as np
import itertools
import math



class Face_Recognition:


    def __init__(
            self,
            face_recognition_model,
            model):

        self.face_recognition_model=face_recognition_model

        self.model=model

    def get_prediction(self,prob, prediction_list_temp, prediction_dict, number_of_count):

        for i in range(0, len(prob)):
            if number_of_count[prediction_list_temp[i]] > 1:
                if prob[i] >= prediction_dict[prediction_list_temp[i]]:
                    self.final_prediction.append(prediction_list_temp[i])
                else:
                    self.final_prediction.append("unknown")
            else:
                # print prediction_list_temp[i]
                # print prob[i]
                if prob[i] > 0.4:
                    self.final_prediction.append(prediction_list_temp[i])
                else:
                    self.final_prediction.append("unknown")

        return True, self.final_prediction

    def get_average_probability(self,prediction, probability, prob):
        count = 1
        for i in range(0, len(prob) - 1):
            if self.actual_prediction[i] == prediction:
                probability = probability + prob[i]
                count = count + 1
        return probability, count

    def get_probability(self,landmark_points, face_encoding_points, model):
        for i in range(0, len(landmark_points)):

            prediction = model.predict_proba(np.array(face_encoding_points[i]).flatten()).ravel()

            max_prob = np.argmax(prediction)

            probability = prediction[max_prob]

            self.actual_probability.append(probability)
            prediction = model.predict(np.array(face_encoding_points[i]).flatten())
            if prediction not in self.actual_prediction:

                self.actual_prediction.append(prediction[0])
            else:
                average_probability, count = self.get_average_probability(prediction, probability, self.actual_probability)

                self.actual_prediction.append(prediction[0])
                average_probability = average_probability / count
            if prediction[0] not in self.average_prediction_dictionary:
                self.average_prediction_dictionary[prediction[0]] = probability
                self.prediction_list.append(prediction[0])
                self.number_of_count[prediction[0]] = 1
            else:
                for i in range(0, len(self.average_prediction_dictionary)):
                    if self.prediction_list[i] == prediction:
                        self.average_prediction_dictionary[prediction[0]] = average_probability
                self.number_of_count[prediction[0]] = self.number_of_count[prediction[0]] + 1
        print self.number_of_count
        return self.actual_probability, self.actual_prediction,self.average_prediction_dictionary, self.number_of_count
    def face_encoding(self,frame,landmark_list):
        return True, [np.array(self.face_recognition_model.compute_face_descriptor(frame,raw_landmark_set)) for raw_landmark_set in landmark_list]

    def Face_Rec(self,frame,landmark_list):
        self.final_prediction = []
        self.actual_prediction = []
        self.prediction_list = []
        self.average_prediction_dictionary = {}
        self.actual_probability = []
        self.number_of_count = {}
        success,face_encoding_points=self.face_encoding(frame,landmark_list)
        prob,prediction_list_temp,prediction_dict,number_of_count=self.get_probability(landmark_list,face_encoding_points,self.model)
        success , prediction_list=self.get_prediction(prob,prediction_list_temp,prediction_dict,number_of_count)
        if success:
            return success, prediction_list







