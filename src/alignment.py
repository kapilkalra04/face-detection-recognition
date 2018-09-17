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
	cnn = load_model('src/CNN_21_1000.h5')
	
	# find the scaling ratios 
	faceHeight = np.float32(face.shape[0])
	faceWidth = np.float32(face.shape[1])
	heightScaling = 96.0/faceHeight
	widthScaling = 96.0/faceWidth
	
	face2 = face
	# resize the image to the size on which the CNN was trained
	faceResized = cv2.resize(face2,(96,96))
	
	# prepare Input for CNN
	faceResized = np.expand_dims(faceResized,axis=0)
	faceResized = np.expand_dims(faceResized,axis=3)
	faceResized = np.float32(faceResized)
	faceResized = faceResized/255.0
	
	# obtain output
	outputVector = cnn.predict(faceResized)
	outputVector = (outputVector*48) + 48

	# scale up the eye centers obtained
	ref_left_eye_center_x = outputVector[0,2]/widthScaling
	ref_left_eye_center_y = outputVector[0,3]/heightScaling
	ref_right_eye_center_x = outputVector[0,0]/widthScaling
	ref_right_eye_center_y = outputVector [0,1]/heightScaling
	print ref_left_eye_center_x,ref_left_eye_center_y,ref_right_eye_center_x,ref_right_eye_center_y
		
	# load haar cascade classifiers
	eye_cascade = cv2.CascadeClassifier('src/haarcascade_eye.xml')
	eye_cascade_2 = cv2.CascadeClassifier('src/haarcascade_eye_2.xml')

    # find eyes using haar cascade for eyes 
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
	
	indexL = 0
	indexR = 0
		
	if(len(eyeCenterLeftX) > 0 ):
		# obtain main left-eye-center through the largest eye-box area criteria
		minimumL = eyeCenterLeftArea[0]
		for i in range(0,len(eyeCenterLeftArea)):
			if eyeCenterLeftArea[i] >= minimumL:
				indexL = i
				minimumL = eyeCenterLeftArea[i]

		# compare obtained haar cordinates to CNN coordinates
		if(abs(eyeCenterLeftX[indexL] - ref_left_eye_center_x) < 2.5/widthScaling):
			 left_eye_center_x = eyeCenterLeftX[indexL]
		else:
			left_eye_center_x = ref_left_eye_center_x
			
		if(abs(eyeCenterLeftY[indexL] - ref_left_eye_center_y) < 2.5/heightScaling):
			 left_eye_center_y = eyeCenterLeftY[indexL]
		else:
			left_eye_center_y = ref_left_eye_center_y
		
	else:
		left_eye_center_x = ref_left_eye_center_x
		left_eye_center_y = ref_right_eye_center_y
		


	if(len(eyeCenterRightX) > 0):
		# obtain main right-eye-center through the largest eye-box area criteria
		minimumR = eyeCenterRightArea[0]
		for i in range(0,len(eyeCenterRightArea)):
			if eyeCenterRightArea[i] >= minimumR:
				indexR = i
				minimumR = eyeCenterRightArea[i]

		# compare obtained haar cordinates to CNN coordinates
		if(abs(eyeCenterRightX[indexR] - ref_right_eye_center_x) < 2.5/widthScaling):
			 right_eye_center_x = eyeCenterRightX[indexR]
		else:
			right_eye_center_x = ref_right_eye_center_x
			
		if(abs(eyeCenterRightY[indexR] - ref_right_eye_center_y) < 2.5/heightScaling):
			 right_eye_center_y = eyeCenterRightY[indexR]
		else:
			right_eye_center_y = ref_right_eye_center_y
		
	else:
		right_eye_center_x = ref_right_eye_center_x
		right_eye_center_y = ref_right_eye_center_y
	
	print left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y	
	return left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y
