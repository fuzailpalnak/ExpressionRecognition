import dlib
import cv2
import numpy as np
import itertools
import math
from collections import OrderedDict

import openface


class FaceDetector:


    def __init__(
            self,
            detector,
            predictor,
            alignedFace,
            desiredLeftEye=(0.35, 0.35),
            desiredFaceWidth=256, desiredFaceHeight=None):

        self.detector = detector
        self.predictor=predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        self.alignedFace=alignedFace
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

        self.FACIAL_LANDMARKS_IDXS = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 36)),
            ("jaw", (0, 17))
        ])



    def detect_dlib(self,frame):
        return frame, self.detector(frame)

    def bbox(self,face,frame):
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return x,y



    def align_head(self,frame,dlib_face, predict=False):

        landmark_list = []
        new_landmark_list=[]
        individual_landmark_list=[]
        detected_landmarks_dlib_format=[]
        coords = np.zeros((68, 2), dtype="int")

        for face_location in dlib_face:
            # face=self.alignedFace.align(534,frame,face_location,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

            detected_landmarks=self.predictor(frame,face_location)
            detected_landmarks_dlib_format.append(detected_landmarks)

            for i in range(0, 68):  # There are 68 landmark points on each face
                landmark_list.append([detected_landmarks.part(i).x, detected_landmarks.part(i).y])
                coords[i] = (detected_landmarks.part(i).x, detected_landmarks.part(i).y)
                individual_landmark_list.append([detected_landmarks.part(i).x, detected_landmarks.part(i).y])


                cv2.circle(frame, (detected_landmarks.part(i).x, detected_landmarks.part(i).y), 1, (0, 10, 255), thickness=2)

            # (lStart, lEnd) = self.FACIAL_LANDMARKS_IDXS["left_eye"]
            # (rStart, rEnd) = self.FACIAL_LANDMARKS_IDXS["right_eye"]
            # leftEyePts = coords[lStart:lEnd]
            # rightEyePts = coords[rStart:rEnd]
            #
            # # compute the center of mass for each eye
            # leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
            # rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
            #
            # # compute the angle between the eye centroids
            # dY = rightEyeCenter[1] - leftEyeCenter[1]
            # dX = rightEyeCenter[0] - leftEyeCenter[0]
            # angle = np.degrees(np.arctan2(dY, dX)) - 180
            #
            # # compute the desired right eye x-coordinate based on the
            # # desired x-coordinate of the left eye
            # desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
            #
            # # determine the scale of the new resulting image by taking
            # # the ratio of the distance between eyes in the *current*
            # # image to the ratio of distance between eyes in the
            # # *desired* image
            # dist = np.sqrt((dX ** 2) + (dY ** 2))
            # desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
            # desiredDist *= self.desiredFaceWidth
            # scale = desiredDist / dist
            #
            # # compute center (x, y)-coordinates (i.e., the median point)
            # # between the two eyes in the input image
            # eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            #               (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
            #
            # # grab the rotation matrix for rotating and scaling the face
            # M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
            #
            # # update the translation component of the matrix
            # tX = self.desiredFaceWidth * 0.5
            # tY = self.desiredFaceHeight * self.desiredLeftEye[1]
            # M[0, 2] += (tX - eyesCenter[0])
            # M[1, 2] += (tY - eyesCenter[1])
            #
            # # apply the affine transformation
            # (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
            # output = cv2.warpAffine(frame, M, (w, h),
            #                         flags=cv2.INTER_CUBIC)
            # cv2.imwrite("temp.png", cv2.cvtColor(face,cv2.COLOR_RGB2BGR))
            # cv2.imwrite("temp1.png", frame)

            new_landmark_list.append(np.asarray(individual_landmark_list).flatten())
            individual_landmark_list=[]
            # print new_landmark_list
            #
            #     xlist.append(float(detected_landmarks.part(i).x))
            #     ylist.append(float(detected_landmarks.part(i).y))
            # xmean = np.mean(xlist)  # Get the mean of both axes to determine centre of gravity
            # ymean = np.mean(ylist)
            # xcentral = [(x - xmean) for x in
            #             xlist]  # get distance between each point and the central point in both axes
            # ycentral = [(y - ymean) for y in ylist]
            # cv2.circle(frame, (int(xmean),int(ymean)), 1, (0, 100, 0), thickness=2)
            # if xlist[26] == xlist[
            #     29]:  # If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
            #     anglenose = 0
            # else:
            #     anglenose = int(math.atan((ylist[26] - ylist[29]) / (xlist[26] - xlist[29])) * 180 / math.pi)
            #
            # if anglenose < 0:
            #     anglenose += 90
            # else:
            #     anglenose -= 90
            #
            # landmarks_vectorised = []
            # for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            #     landmarks_vectorised.append(x)
            #     landmarks_vectorised.append(y)
            #     meannp = np.asarray((ymean, xmean))
            #     coornp = np.asarray((z, w))
            #     dist = np.linalg.norm(coornp - meannp)
            #     anglerelative = (math.atan((z - ymean) / (w - xmean)) * 180 / math.pi) - anglenose
            #     landmarks_vectorised.append(dist)
            #     landmarks_vectorised.append(anglerelative)
        # return True , np.array(landmarks_vectorised)
        # print np.asarray(new_landmark_list)
        # print np.asarray(new_landmark_list).shape

        if predict:
            return True,np.asarray(new_landmark_list),detected_landmarks_dlib_format
        else:

            return True , np.array(landmark_list)



