#!/usr/bin/python3

import numpy as np
import cv2
from matplotlib import pyplot as plt

img_path = "sat_img_stavanger_bottom.png"
img_path2 = "google_stavanger_bottom.png"
# img_path = "SatelliteImage.png"
# img_path2 = "GeoreferencedImage.png"

# sift = cv2.xfeatures2d.SURF_create(800)
sift = cv2.xfeatures2d.SIFT_create(800)

flann_index = 1
flann_parameters = dict(algorithm = flann_index, trees = 5)
img_matcher = cv2.FlannBasedMatcher(flann_parameters, {})

image1 = cv2.imread(img_path)
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
kpts1, descs1 = sift.detectAndCompute(gray_image1,None)

image2 = cv2.imread(img_path2)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
kpts2, descs2 = sift.detectAndCompute(gray_image2,None)

img_matches = img_matcher.knnMatch(descs1, descs2, 2)

img_matchesMask = [[0,0] for i in range(len(img_matches))]

for i, (m1,m2) in enumerate(img_matches):
	if m1.distance < 0.45 * m2.distance:
		img_matchesMask[i] = [1,0]
		pt1 = kpts1[m1.queryIdx].pt
		pt2 = kpts2[m1.trainIdx].pt
		print(i, pt1,pt2 )
		if i % 5 ==0:
			cv2.circle(image1, (int(pt1[0]),int(pt1[1])), 5, (255,0,255), -1)
			cv2.circle(image2, (int(pt2[0]),int(pt2[1])), 5, (255,0,255), -1)

draw_params = dict(matchColor = (0, 255,0),
				  singlePointColor = (0,0,255),
		  		  matchesMask = img_matchesMask,
				  flags = 0)

res = cv2.drawMatchesKnn(image1,kpts1,image2,kpts2,img_matches,None,**draw_params)

res = cv2.resize(res, (1080, 720))

cv2.imshow("Result", res);cv2.waitKey();cv2.destroyAllWindows()
