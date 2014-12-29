import cv2
import facedetect as fdmod
import hashlib
import json
import numpy as np
import os
import phash
import sys
sys.path.append('/home/anand/Downloads/devbox_configs/')
import backend
import tornado
import tornado.httpserver
from tornado.options import define, options
from tornado.web import RequestHandler, Application

define('debug', default=1, help='hot deployment. use in dev only', type=int)
define('port', default=8888, help='run on the given port', type=int)
define('imgFolder', default='uploadedImages', help = 'folder to store uploaded images', type=str)

class FaceDetectHandler(RequestHandler):
    def get(self):
        return self.render('imageupload.html')

    def post(self):
        assert(self.request.files.get('file_inp'))
        compareFaces = bool(self.request.files.get('file_inp1'))
        # Read the images
        imgBytes = self.request.files.get('file_inp')[0].body
        imgFileName = self.request.files.get('file_inp')[0].filename
        imgHash = hashlib.sha512(imgBytes).hexdigest()
        imgNew = not backend.redisToGoConn.sismember(fdmod.FACEDETECT_IMG_HASHES, imgHash)
        imgNpArr = np.fromstring(imgBytes, np.uint8)
        imgType = imgFileName.split('.')[1]
        imgType = '.jpg'
        image = cv2.imdecode(imgNpArr, cv2.CV_LOAD_IMAGE_COLOR)
        imgPath = os.path.join(options.imgFolder, imgHash + imgType)
        imgNew = True
        if imgNew:
            cv2.imwrite(imgPath, image)
            backend.redisToGoConn.sadd(fdmod.FACEDETECT_IMG_HASHES, imgHash)
            backend.redisToGoConn.pfadd(fdmod.FACEDETECT_IMG_CNT)
        if compareFaces:
            imgBytes1 = self.request.files.get('file_inp1')[0].body
            imgFileName1 = self.request.files.get('file_inp1')[0].filename
            imgHash1 = hashlib.sha512(imgBytes1).hexdigest()
            imgNew1 = not backend.redisToGoConn.sismember(fdmod.FACEDETECT_IMG_HASHES, imgHash1)
            imgNpArr1 = np.fromstring(imgBytes1, np.uint8)
            imgType1 = imgFileName1.split('.')[1]
            image1 = cv2.imdecode(imgNpArr1, cv2.CV_LOAD_IMAGE_COLOR)
            imgPath1 = os.path.join(options.imgFolder, imgHash1 + imgType1)
            if imgNew1:
                cv2.imwrite(imgPath1, image1)
                backend.redisToGoConn.sadd(fdmod.FACEDETECT_IMG_HASHES, imgHash1)
                backend.redisToGoConn.pfadd(fdmod.FACEDETECT_IMG_CNT)
            same_face = bool(phash.cross_correlation(imgHash,imgHash1))
            self.finish(json.dumps({"sameface": same_face}))
        else:
            FD = fdmod.FeatureDetect(image)
            FD.detectFace()
            FD.detectEyes()
            FD.detectLips()
            self.render('areaselect.html', features=FD.features, imgPath=imgPath)
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

class Application(Application):
    def __init__(self):
        handlers = [
                (r'/', FaceDetectHandler),
                ]
        settings = dict(
            autoescape=None,  # tornado 2.1 backward compatibility
            debug=options.debug,
            gzip=True,
            xheaders=True,

            )
        settings.update({'static_path':os.path.join(os.path.dirname(__file__), 'static')})
        settings.update({'template_path': os.path.join(os.path.dirname(__file__), 'static', 'html')})
        tornado.web.Application.__init__(self, handlers, **settings)
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'static', options.imgFolder)):
            os.makedirs(os.path.join(os.path.dirname(__file__), 'static', options.imgFolder))

def main():
    tornado.options.parse_command_line()
    App = Application()
    httpserver = tornado.httpserver.HTTPServer(App)
    httpserver.listen(port=options.port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()
