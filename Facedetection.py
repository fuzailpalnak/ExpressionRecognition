import dlib
import cv2
import numpy as np
import itertools
import math



class FaceDetector:


    def __init__(
            self,
            face_casc='params/haarcascade_frontalface_default.xml',
            scale_factor=4):

        self.scale_factor = scale_factor
        # load pre-trained cascades
        self.face_casc = cv2.CascadeClassifier(face_casc)
        if self.face_casc.empty():
            print 'Warning: Could not load face cascade:', face_casc
            raise SystemExit


    def detect(self, frame):

        frameCasc = cv2.cvtColor(
            cv2.resize(
                frame,
                (0, 0),
                fx=1.0 / self.scale_factor,
                fy=1.0 / self.scale_factor),
            cv2.COLOR_RGB2GRAY)
        #detector = dlib.get_frontal_face_detector()
        faces = self.face_casc.detectMultiScale(
            frameCasc,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=cv2.CASCADE_SCALE_IMAGE) * self.scale_factor

        # if face is found: extract head region from bounding box
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            head = cv2.cvtColor(frame[y:y + h, x:x + w],
                                cv2.COLOR_RGB2GRAY)
            dlib_rect=dlib.rectangle(x.astype(long), y.astype(long), (x + w).astype(long), (y + h).astype(long))


            return True, frame, head, dlib_rect,(x,y)

        return False, frame, None,None, (0, 0)

    def align_head(self, head,dlib_rect,predictor,frame):

        landmark_list = []
        xlist=[]
        ylist=[]

        detected_landmarks = predictor(frame, dlib_rect).parts()


        landmark_list = np.matrix([[p.x, p.y] for p in detected_landmarks])

        for k, d in enumerate(landmark_list):  # For each detected face
            pos = (d[0, 0], d[0, 1])


            cv2.circle(frame, pos, 1, color=(0, 0, 255))

          #to show landmarks during training

        #cv2.imshow("s", frame)
        #gray = cv2.cvtColor(
         #   frame,
          #  cv2.COLOR_RGB2GRAY)
        #cv2.imwrite("facelandmark.png", gray)
        return True , np.array(landmark_list)



