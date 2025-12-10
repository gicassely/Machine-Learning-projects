#! /usr/bin/env python
import sys
from skimage import feature
import cv2
import numpy as np


def featureExtractor(imagePath, nPoints, radius):
    img = cv2.imread(imagePath)
    #print imagePath
    #print img.shape
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lbp = feature.local_binary_pattern(grayImg, nPoints, radius, method="uniform")
    hist = np.histogram(lbp.ravel(), bins=np.arange(0, nPoints + 3), range=(0, nPoints + 2), density = True)[0]

    return(hist)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: %s <image path>' % sys.argv[0])
        sys.exit()

    image = sys.argv[1]
    nPoints = 24
    radius = 8

    featureVector = featureExtractor(image, nPoints, radius)
    #print ' '.join(map(str, featureVector))
    print ((str, featureVector))
