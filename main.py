#!/usr/bin/env python3

import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt


def cornerHarris():
    filename = sys.argv[1]
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def shi_tomasi():
    filename = sys.argv[1]
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray,500,0.01,10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)

    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def canny_edge():
    filename = sys.argv[1]

    img = cv2.imread(filename,0)

    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    print(laplacian)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()

def gaussianBlur():
    filename = sys.argv[1]

    img = cv2.imread(filename, 0)

    src = cv2.GaussianBlur(img, (11, 11), 0)

    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(src,cmap = 'gray')
    plt.title('Blur'), plt.xticks([]), plt.yticks([])

    plt.show()


def _orb():
    filename = sys.argv[1]

    img = cv2.imread(filename)

    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    #img2 = cv2.drawKeypoints(img, kp, color=(0,255,0), flags=0)
    for i in kp:
        cv2.circle(img,(int(i.pt[0]), int(i.pt[1])),3,255,-1)

    plt.imshow(img)
    plt.show()


_orb()