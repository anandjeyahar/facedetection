import cv2
import hashlib
import json
import numpy as np
import os
from PIL import Image
import sys
import tornado
from tornado.options import define, options
from tornado.web import RequestHandler, Application

# Custom backend account settings
sys.path.append("/home/anand/Downloads/devbox_configs/")
import backend

define('debug', default=1, help='hot deployment. use in dev only', type=int)
define('port', default=8000, help='run on the given port', type=int)
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

class Application(Application):
    #  """
    #  >>> import requests
    #  >>> requests.post("/shorten", params={"orig_url":"http://google.com"})
    #  >>> resp = requests.get("/shorten", params={"short_url": "265477614567132497141480353139365708304L"})
    #  >>> assert resp.url=="http://google.com"
    #  """
    def __init__(self):
        handlers = [
                (r'/facedetect', FaceDetectHandler),
                ]
        settings = dict(
            autoescape=None,  # tornado 2.1 backward compatibility
            debug=options.debug,
            gzip=True,
            )
        settings.update({'static_path':'./static'})
        tornado.web.Application.__init__(self, handlers, **settings)
        if not os.path.exists(options.imgFolder):
            os.makedirs(options.imgFolder)

class FaceDetectHandler(RequestHandler):
    def get(self):
        self.render('static/imageupload.html')

    def post(self, image=None):
        # Read the image
        imgBytes = self.request.files.get('file_inp')[0].body
        imgFileName = self.request.files.get('file_inp')[0].filename
        imgHash = hashlib.sha512(imgBytes).hexdigest()
        imgNew = not backend.redisConn.sismember(SET_IMG_HASHES, imgHash)
        #backend.redislabs.pfadd(FACEDETECT_IMG_HASHES)
        imgNparr = np.fromstring(imgBytes, np.uint8)
        imgType = imgFileName.split('.')[1]
        imgType = '.jpg'
        image = cv2.imdecode(imgNparr, cv2.CV_LOAD_IMAGE_COLOR)
        if imgNew:
            cv2.imwrite(os.path.join(options.imgFolder, imgHash + imgType), image)
            backend.redisConn.sadd(SET_IMG_HASHES, imgHash)
        FD = FeatureDetect(image)
        FD.detectFace()
        FD.detectEyes()
        FD.detectLips()
        self.finish(json.dumps(FD.features))
        # Draw a rectangle around the faces
        # print "Found {0} faces!".format(len(FD.faces))
        # for (x, y, w, h) in FD.faces:
        #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # print "Found {0} eyes".format(len(FD.eyes))
        # for (x, y, w, h) in FD.eyes:
        #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # print "Found {0} lips".format(len(FD.lips))
        # for (x, y, w, h) in FD.lips:
        #     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

def main():
    tornado.options.parse_command_line()
    app = Application()

    app.listen(options.port, xheaders=True)
    loop = tornado.ioloop.IOLoop.instance()
    loop.start()

if __name__ == "__main__":
    main()
