import facedetect as fdmod
import hotshot
import cv2

def detectFace():
    image = cv2.imread('./abba.png')
    FD = fdmod.FeatureDetect(image)
    FD.detectFace()

if __name__ == '__main__':
    prof = hotshot.Profile('haarcascade.prof')
    benchtime, stones = prof.runcall(detectFace)
    prof.close()

    stats = hotshot.stats.load('haarcascade.prof')
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats(20)
