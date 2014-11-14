import bottle
from tornado.options import define, options
from tornado.web import RequestHandler, Application

define('debug', default=1, help='hot deployment. use in dev only', type=int)
define('port', default=8000, help='run on the given port', type=int)

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
        imgPath = os.path.join(options.imgFolder, imgHash + imgType)
        if imgNew:
            cv2.imwrite(imgPath, image)
            backend.redisConn.sadd(SET_IMG_HASHES, imgHash)
        FD = FeatureDetect(image)
        FD.detectFace()
        FD.detectEyes()
        FD.detectLips()
        self.render('static/areaselect.html',imgPath=imgPath )
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

def main():
    tornado.options.parse_command_line()
    app = Application()

    app.listen(options.port, xheaders=True)
    loop = tornado.ioloop.IOLoop.instance()
    loop.start()

if __name__ == "__main__":
    main()
