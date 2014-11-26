import cv2
import sys
# Custom backend account settings
sys.path.append("/home/anand/Downloads/devbox_configs/")
import backend
import facedetect as fdmod
import flask
import os
import tornado
from tornado.options import define, options
from tornado.web import RequestHandler, Application

define('debug', default=1, help='hot deployment. use in dev only', type=int)
define('port', default=8888, help='run on the given port', type=int)
define('imgFolder', default='./uploadedImages', help = 'folder to store uploaded images', type=str)

class FaceDetectHandler(RequestHandler):
    def get(self):
        return self.render('static/imageupload.html')

    def post(self):
        # Read the image
        request = self.request
        imgFd = request.files.get('file_inp')
        imgFileName = imgFd.filename
        imgName, imgType = imgFileName.split('.')
        imgType = '.jpg'
        imgFullPath = os.path.join(options.imgFolder, imgName + imgType)
        imgFd.save(imgFullPath)
        # imgHash = hashlib.sha512(imgBytes).hexdigest()
        # imgNew = not backend.redisConn.sismember(fdmod.SET_IMG_HASHES, imgHash)
        # backend.redislabs.pfadd(fdmod.SET_IMG_HASHES)
        image = cv2.imread(imgFullPath)
        imgNew = True
        if imgNew:
            cv2.imwrite(imgFullPath, image)
            #backend.redisConn.sadd(fdmod.SET_IMG_HASHES, imgHash)
        FD = fdmod.FeatureDetect(image)
        FD.detectFace()
        import pdb; pdb.set_trace()
        FD.detectEyes()
        FD.detectLips()
        return self.render('static/areaselect.html',imgPath=imgFullPath )
        #self.finish(json.dumps(FD.features))
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
    #  """
    #  >>> import requests
    #  >>> requests.post("/shorten", params={"orig_url":"http://google.com"})
    #  >>> resp = requests.get("/shorten", params={"short_url": "265477614567132497141480353139365708304L"})
    #  >>> assert resp.url=="http://google.com"
    #  """
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
        tornado.web.Application.__init__(self, handlers, **settings)
        if not os.path.exists(options.imgFolder):
            os.makedirs(options.imgFolder)

def main():
    tornado.options.parse_command_line()
    App = Application()
    App.listen(address='0.0.0.0', port=options.port, xheaders=True)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()
