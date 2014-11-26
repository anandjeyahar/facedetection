import cv2
import sys
# Custom backend account settings
sys.path.append("/home/anand/Downloads/devbox_configs/")
import backend
import facedetect as fdmod
import json
import hashlib
import numpy as np
import os
import tornado
from tornado.options import define, options
from tornado.web import RequestHandler, Application
redistogo_url = os.getenv('REDISTOGOURL')
if redis_url:
    #redis_url = redistogo_url.split('redis://redistogo:')[1]
    #redis_url = redis_url.split('/')[0]
    redisToGoConn = redis.from_url(redistogo_url)

define('debug', default=1, help='hot deployment. use in dev only', type=int)
define('port', default=8888, help='run on the given port', type=int)
define('imgFolder', default='uploadedImages', help = 'folder to store uploaded images', type=str)

class FaceDetectHandler(RequestHandler):
    def get(self):
        return self.render('imageupload.html')

    def post(self):
        # Read the image
        imgBytes = self.request.files.get('file_inp')[0].body
        imgFileName = self.request.files.get('file_inp')[0].filename
        imgHash = hashlib.sha512(imgBytes).hexdigest()
        imgNew = not redisToGoConn.sismember(fdmod.FACEDETECT_IMG_HASHES, imgHash)
        imgNparr = np.fromstring(imgBytes, np.uint8)
        imgType = imgFileName.split('.')[1]
        imgType = '.jpg'
        image = cv2.imdecode(imgNparr, cv2.CV_LOAD_IMAGE_COLOR)
        imgPath = os.path.join(options.imgFolder, imgHash + imgType)
        imgNew = True
        if imgNew:
            cv2.imwrite(imgPath, image)
            redisToGoConn.sadd(fdmod.FACEDETECT_IMG_HASHES, imgHash)
            redisToGoConn.pfadd(fdmod.FACEDETECT_IMG_CNT)
        FD = fdmod.FeatureDetect(image)
        FD.detectFace()
        FD.detectEyes()
        FD.detectLips()
        #return self.render('areaselect.html',imgPath=imgPath, header_text="Play around, have fun" )
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

class Application(Application):
    def __init__(self):
        handlers = [
                (r'/', FaceDetectHandler),
                ]
        settings = dict(
            autoescape=None,  # tornado 2.1 backward compatibility
            debug=options.debug,
            gzip=True,
            )
        settings.update({'static_path':'./static'})
        settings.update({'template_path': os.path.join(os.path.dirname(__file__), 'static', 'html')})
        tornado.web.Application.__init__(self, handlers, **settings)
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'static', options.imgFolder):
            os.makedirs(options.imgFolder)

def main():
    tornado.options.parse_command_line()
    App = Application()
    App.listen(address='0.0.0.0', port=options.port, xheaders=True)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()
