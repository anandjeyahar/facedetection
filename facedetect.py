import cv2
import hashlib
import json
import numpy as np
import os
from PIL import Image
import sys
import tornado
from tornado.options import define, options

FACEDETECT_IMG_CNT = 'facedetect:img:cnt'
FACEDETECT_IMG_HASHES = 'facedetect:img:hashes'
CLASSIFICATIONS = ['opencv-haarcascade', 'HP-IDOL']

class TooManyFacesException(Exception):
    pass

class FeatureDetect(object):
    self.HpIDOLOnDemandAsyncUrl = 'https://api.idolondemand.com/1/api/async/detectfaces/v1'
    self.HpIDOLOnDemandSyncUrl = 'https://api.idolondemand.com/1/api/sync/detectfaces/v1'
    def __init__(self, image, classifier=None):
        assert classifier in CLASSIFICATIONS, logger.error('unknown classifier')
        self.classifier = CLASSIFICATIONS[0]
        if classifier == CLASSIFICATIONS[0]:
            # TODO: add support for multiple training files and going through them one by one.
            self.frontalFaceCascPath = "haarcascade_frontalface_default.xml"
            self.mouthCascPath = "Mouth.xml"
            self.noseCascPath = "Nariz.xml"
            self.eyeCascPath = "haarcascades/haarcascade_eye.xml"
            # Create the haar cascade
            self.faceCascade = cv2.CascadeClassifier(self.frontalFaceCascPath)
            self.mouthCascade = cv2.CascadeClassifier(self.mouthCascPath)
            self.eyeCascade = cv2.CascadeClassifier(self.eyeCascPath)
            self.noseCascade = cv2.CascadeClassifier(self.noseCascPath)
        self.image = image
        self.grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.features = dict()

    def detectFace(self):
        # Detect faces in the image
        if self.classifier == CLASSIFICATIONS[0]:
            self.faces = self.faceCascade.detectMultiScale(
                self.grayImage,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(30, 30),
                flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                )
        else:
            args = { 'apikey': backend.topCoderHpIDOLOnDemandApiKey,
                     'text': 'text',
                     'highlight_expression': 'link',

                     }
            resp = request.get(self.HpIDOLOnDemandSyncUrl, args)
            logger.info(resp)
        if len(self.faces) > 0:
            self.features.update({"faceCorners":str(self.faces[0])})
        else:
            self.features.update({"faceCorners":None})

    def detectNose(self):
        self.noses = self.noseCascade.detectMultiScale(
            self.grayImage,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(25, 15),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
#        assert len(self.noses) == 1, raise TooManyFacesException({"noses":self.noses})
        if len(self.noses) > 0:
            self.features.update({"noseCorners":str(self.noses[0])})
        else:
            self.features.update({"noseCorners":None})

    def detectLips(self):
        # Detect faces in the image
        self.lips = self.faceCascade.detectMultiScale(
            self.grayImage,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(25, 15),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
#        assert len(self.lips) == 1, raise TooManyFacesException({"faces":self.lips})
        if len(self.lips) > 0:
            self.features.update({"lipCorners":str(self.lips[0])})
        else:
            self.features.update({"lipCorners":None})

    def detectEyes(self):
        # Detect faces in the image
        self.eyes = self.eyeCascade.detectMultiScale(
            self.grayImage,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(30, 30),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
#        assert len(self.eyes) == 1, raise TooManyFacesException({"faces":self.eyes})
        if len(self.eyes) > 0:
            self.features.update({"eyeCorners":str(self.eyes[0])})
        else:
            self.features.update({"eyeCorners":None})


