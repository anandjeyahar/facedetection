import cv2
import sys

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
        self.features = {"faceCorners":self.faces[0]}

    def detectNose(self):
        self.noses = self.noseCascade.detectMultiScale(
            self.grayImage,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(30, 30),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
#        assert len(self.noses) == 1, raise TooManyFacesException({"noses":self.noses})
        self.features = {"noseCorners":self.noses[0]}

    def detectLips(self):
        # Detect faces in the image
        self.lips = self.faceCascade.detectMultiScale(
            self.grayImage,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(30, 30),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
#        assert len(self.lips) == 1, raise TooManyFacesException({"faces":self.lips})
        self.features = {"lipCorners":self.lips[0]}

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
        self.features = {"eyeCorners":self.eyes[0]}

def main():
    # Get user supplied values
    imagePath = sys.argv[1]
    # Read the image
    image = cv2.imread(imagePath)
    FD = FeatureDetect(image)
    FD.detectFace()
    FD.detectEyes()
    FD.detectLips()

    # Draw a rectangle around the faces
    print "Found {0} faces!".format(len(FD.faces))
    for (x, y, w, h) in FD.faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in FD.eyes:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for (x, y, w, h) in FD.lips:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
