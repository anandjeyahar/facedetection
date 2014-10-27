import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
class TooManyFacesException(Exception):
    pass


class FeatureDetect(self):
    def __init__(self, image):
        frontalFaceCascPath = "haarcascade_frontalface_default.xml"
        # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(frontalFaceCascPath)

    def detectFace(self):
        # Read the image
        image = cv2.imread(imagePath)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            grayImage,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(30, 30),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        assert len(faces) == 1, raise TooManyFacesException({"faces":faces})
        self.features = {"faceCorners":(x,y,w,h)}
        print "Found {0} faces!".format(len(faces))

    def detectEyes(self):
        pass

    def detectLips(self):
        pass

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
