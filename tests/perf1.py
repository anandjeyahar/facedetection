
def main():
    import timeit
    import facedetect as fdmod
    import cv2
    image = cv2.imread('./abba.png')
    FD = fdmod.FeatureDetect(image)
    FD.detectFace()

if __name__ == '__main__':
    import timeit
    print timeit.timeit('main()', setup='from __main__ import main',number=10)
