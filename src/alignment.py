import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd
 
def rotate(face,left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y):
	lx = left_eye_center_x
	ly = left_eye_center_y
	rx = right_eye_center_x
	ry = right_eye_center_y

	# 5/4 ratio
	# desiredHeight = 512
	# desiredWidth = 409	
	# desiredLeftEye = 123
	# desiredRightEye = 286

	# carry out angle calculations through arctan
	dY = ry - ly                                       
	dX = rx - lx

	angle = np.degrees(np.arctan2(dY, dX))				# angle should be in degrees
	
	scale = 1										

	cx = (rx+lx)/2.0
	cy = (ry+ly)/2.0
	center = (cx,cy)									# rotation will take place around the eye center
	
	return center, angle, scale

def detectEyeCenters(face):
	# cnn = load_model('src/cnn70.h5')
	
	# # find the scaling ratios 
	# faceHeight = np.float32(face.shape[0])
	# faceWidth = np.float32(face.shape[1])
	# heightScaling = 96.0/faceHeight
	# widthScaling = 96.0/faceWidth
	
	# faceResized = cv2.resize(face,(96,96))
	# faceResized = cv2.fastNlMeansDenoising(faceResized)
	# plt.show()

	# plt.imshow(faceResized,cmap='gray')
	
	# # prepare Input for CNN
	# faceResized = np.expand_dims(faceResized,axis=0)
	# faceResized = np.expand_dims(faceResized,axis=3)
	# faceResized = np.float32(faceResized)
	# faceResized = faceResized/255.0
	
	# # obtain output
	# outputVector = cnn.predict(faceResized)
	# outputVector = (outputVector*48) + 48
	
	# X=[]
	# Y=[]
	# for i in range(0,30):
	# 	if i%2 == 0:
	# 		X.append(outputVector[0,i])
	# 	else:
	# 		Y.append(outputVector[0,i])
	# plt.plot(X,Y,'*',markersize=3)
	# plt.show()
	# print outputVector

	# # scale up the eye centers obtained
	# ref_left_eye_center_x = outputVector[0,0]/widthScaling
	# ref_left_eye_center_y = outputVector[0,1]/heightScaling
	# ref_right_eye_center_x = outputVector[0,2]/widthScaling
	# ref_right_eye_center_y = outputVector [0,3]/heightScaling
	
	


	# load haar cascade classifiers
	eye_cascade = cv2.CascadeClassifier('src/haarcascade_eye.xml')
	eye_cascade_2 = cv2.CascadeClassifier('src/haarcascade_eye_2.xml')

	eyes = eye_cascade.detectMultiScale(face)
	
	if(len(eyes)<2):
		eyes = eye_cascade_2.detectMultiScale(face)

	print eyes	
	
	boundaryX = face.shape[1]/2.0			# separate them into Left and Right
	boundaryY = face.shape[0]/2.0			# remove bottom half false candidates

	eyeCenterLeftX = []						
	eyeCenterLeftY = []
	eyeCenterLeftArea = []
	
	eyeCenterRightX = []
	eyeCenterRightY = []
	eyeCenterRightArea = []
	
	# separate out all possible eye centers candidate into LHS and RHS candidates
	for i in range(0,len(eyes)):
		if(eyes[i][0] + (eyes[i][2]/2.0) <= boundaryX - (boundaryX/16) and eyes[i][1] + (eyes[i][3]/2.0) <= boundaryY):
			eyeCenterLeftX.append(eyes[i][0] + (eyes[i][2]/2.0))
			eyeCenterLeftY.append(eyes[i][1] + (eyes[i][3]/2.0))
			eyeCenterLeftArea.append(eyes[i][2] * eyes[i][3])
		if(eyes[i][0] + (eyes[i][2]/2.0) > boundaryX + (boundaryX/16) and eyes[i][1] + (eyes[i][3]/2.0) <= boundaryY):
			eyeCenterRightX.append(eyes[i][0] + (eyes[i][2]/2.0))
			eyeCenterRightY.append(eyes[i][1] + (eyes[i][3]/2.0))
			eyeCenterRightArea.append(eyes[i][2] * eyes[i][3])
	
	# obtain main left-eye-center and right-eye-center through largest area criteria
	indexL = 0
	indexR = 0
	minimumL = eyeCenterLeftArea[0]
	minimumR = eyeCenterRightArea[0]
	for i in range(0,len(eyeCenterLeftArea)):
		if eyeCenterLeftArea[i] >= minimumL:
			indexL = i
			minimumL = eyeCenterLeftArea[i]
	for i in range(0,len(eyeCenterRightArea)):
		if eyeCenterRightArea[i] >= minimumR:
			indexR = i
			minimumR = eyeCenterRightArea[i]
	
	left_eye_center_x = eyeCenterLeftX[indexL]
	left_eye_center_y = eyeCenterLeftY[indexL]
	right_eye_center_x = eyeCenterRightX[indexR]
	right_eye_center_y = eyeCenterRightY[indexR]

	X=[]
	Y=[]
	X.extend([left_eye_center_x,right_eye_center_x])
	Y.extend([left_eye_center_y,right_eye_center_y])
	# plt.plot(X,Y,'*',markersize=3)
	# plt.show()
	

	# print ref_left_eye_center_x,ref_left_eye_center_y,ref_right_eye_center_x,ref_right_eye_center_y
	print left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y	
	return left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y
