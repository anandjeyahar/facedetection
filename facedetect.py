import cv2
import hashlib
import json
import numpy as np
import os
from PIL import Image
import sys
import tornado
from tornado.options import define, options
# Custom backend account settings
import backend

define('imgFolder', default='./uploadedImages', help = 'folder to store uploaded images', type=str)

# FACEDETECT_IMG_HASHES = 'facedetect:img:hashes'
SET_IMG_HASHES = 'set:img:hashes'

class TooManyFacesException(Exception):
    pass

class FeatureDetect(object):
    def __init__(self, image):
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
        self.faces = self.faceCascade.detectMultiScale(
            self.grayImage,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(30, 30),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
#        assert len(faces) == 1, raise TooManyFacesException({"faces":faces})
        self.features.update({"faceCorners":str(self.faces[0])})

    def detectNose(self):
        self.noses = self.noseCascade.detectMultiScale(
            self.grayImage,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(25, 15),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
#        assert len(self.noses) == 1, raise TooManyFacesException({"noses":self.noses})
        self.features.update({"noseCorners":str(self.noses[0])})

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
        self.features.update({"lipCorners":str(self.lips[0])})

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
        self.features.update({"eyeCorners":str(self.eyes[0])})



