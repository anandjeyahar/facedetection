import cv2
import hashlib
import json
import numpy as np
import os
#from pillow import Image
from PIL import Image
import sys
import tornado
import logging
import operator
from nude import Nude
from functools import partial
from tornado.options import define, options

logger = logging.getLogger(__name__)

FACEDETECT_IMG_CNT = 'facedetect:img:cnt'
FACEDETECT_IMG_HASHES = 'facedetect:img:hashes'
CLASSIFICATIONS = ['opencv-haarcascade', 'HP-IDOL']

CASCADE_PATH = partial(os.path.join, '/home/anand/Downloads/devbox_configs/facedetect_app/haarcascades')

class FeatureDetect(object):
    def __init__(self, image=None, classifier='opencv-haarcascade'):
        self.HpIDOLOnDemandAsyncUrl = 'https://api.idolondemand.com/1/api/async/detectfaces/v1'
        self.HpIDOLOnDemandSyncUrl = 'https://api.idolondemand.com/1/api/sync/detectfaces/v1'
        assert classifier in CLASSIFICATIONS, logger.error('unknown classifier')
        self.classifier = CLASSIFICATIONS[0]
        if classifier == CLASSIFICATIONS[0]:
            # TODO: add support for multiple training files and going through them one by one.
            self.frontalFaceCascPath = map(CASCADE_PATH, ['haarcascade_frontalface_default.xml',
                                                          # 'haarcascade_frontalface_alt2.xml',
                                                          # 'haarcascade_frontalface_alt_tree.xml',
                                                          # 'haarcascade_frontalface_alt.xml'
                                                          ])
            self.mouthCascPath = map(CASCADE_PATH, ['Mouth.xml',
                                                    # 'haarcascade_smile.xml'
                                                    ])
            self.noseCascPath = map(CASCADE_PATH, ['Nariz.xml'])
            self.eyeCascPath = map(CASCADE_PATH, ['haarcascade_eye.xml',
                                                # 'haarcascade_eye_tree_eyeglasses.xml',
                                                # 'haarcascade_righteye_2splits.xml',
                                                # 'haarcascade_lefteye_2splits.xml',
                                                ])
            # Create the haar cascade
            self.faceCascade = map(cv2.CascadeClassifier, self.frontalFaceCascPath)
            self.mouthCascade = map(cv2.CascadeClassifier, self.mouthCascPath)
            self.eyeCascade = map(cv2.CascadeClassifier, self.eyeCascPath)
            self.noseCascade = map(cv2.CascadeClassifier, self.noseCascPath)
        self.image = image
        self.grayImage = None
        if image is not None:
            self.grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.features = dict()

    def detectFace(self):
        assert self.grayImage is not None
        # Detect faces in the image
        if self.classifier == CLASSIFICATIONS[0]:
            for casc in self.faceCascade:
                self.faces = casc.detectMultiScale(
                    self.grayImage,
                    scaleFactor=1.1,
                    minNeighbors=2,
                    minSize=(30, 30),
                    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                    )
                if len(self.faces) > 0 and all(map(lambda x:x>=0, self.faces[0])):
                    break
        else:
            args = { 'apikey': backend.topCoderHpIDOLOnDemandApiKey,
                     'text': 'text',
                     'highlight_expression': 'link',

                     }
            resp = request.get(self.HpIDOLOnDemandSyncUrl, args)
            logger.info(resp)
        if len(self.faces) > 0:
            self.features.update({'faceCorners':self.faces[0].tolist()})
        else:
            self.features.update({'faceCorners':None})

    def detectNose(self):
        assert self.grayImage is not None
        for casc in self.noseCascade:
            self.noses = casc.detectMultiScale(
                 self.grayImage,
                 scaleFactor=1.1,
                 minNeighbors=2,
                 minSize=(25, 15),
                 flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            if len(self.noses) > 0 and all(map(lambda x:x>=0, self.noses[0])):
                break
        if len(self.noses) > 0:
            self.features.update({'noseCorners':self.noses[0].tolist()})
        else:
            self.features.update({'noseCorners':None})

    def detectLips(self):
        assert self.grayImage is not None
        # Detect faces in the image
        for casc in self.mouthCascade:
            self.lips = casc.detectMultiScale(
                self.grayImage,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(25, 15),
                flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            if len(self.lips) > 0 and all(map(lambda x:x>=0, self.lips[0])):
                break
        if len(self.lips) > 0:
            self.features.update({'lipCorners':self.lips[0].tolist()})
        else:
            self.features.update({'lipCorners':None})

    def detectEyes(self):
        assert self.grayImage is not None
        for casc in self.eyeCascade:
            # Detect faces in the image
            self.eyes = casc.detectMultiScale(
                self.grayImage,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(30, 30),
                flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            if len(self.eyes) > 0 and all(map(lambda x:x>=0, self.eyes[0].tolist())):
                break

        if len(self.eyes) > 0:
            self.features.update({'eyeCorners':self.eyes[0].tolist()})
        else:
            self.features.update({'eyeCorners':None})

    def detectNudeAreas(self):
        assert self.grayImage is not None
        # Damn it this was a waste of time. this detects
        # skin based on colour and under only certain lighting conditions
        n = Nude(Image.fromarray(self.image))
        n.parse()
        self.features.update({'skinAreas':getSkinRegionCoordinates(n.detected_regions, self.image.shape)})

def getSkinRegionCoordinates(detectedRegions, imgSize):
    xlist = list()
    ylist = list()
    for regions in detectedRegions:
        if regions:
            xlist.extend(map(lambda skin: skin.x, regions))
            ylist.extend(map(lambda skin: skin.y, regions))

    return (min(xlist if xlist else [0]),
            min(ylist if ylist else [0]),
            max(xlist if xlist else [imgSize[0]]),
            max(ylist if ylist else [imgSize[1]]))
