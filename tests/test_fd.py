import facedetect
import cv2
def test_fd():
    image = cv2.imread('me.jpg')
    print image.shape
    FD = facedetect.FeatureDetect(image)
    FD.detectEyes()
    FD.detectFace()
    print FD.features

if __name__ == '__main__':
    test_fd()

