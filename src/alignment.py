import cv2
import numpy as np
 
def rotate(face,left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y):
	lx = left_eye_center_x
	ly = left_eye_center_y
	rx = right_eye_center_x
	ry = right_eye_center_y

	# 5/4 ration
	desiredHeight = 512
	desiredWidth = 409	
	desiredLeftEye = 123
	desiredRightEye = 286

	dY = ry - ly
	dX = rx - lx

	angle = np.degrees(np.arctan2(dY, dX))				# angle should be in degrees
	
	scale = 1

	cx = (rx+lx)//2
	cy = (ry+ly)//2
	center = (cx,cy)
	
	return center, angle, scale

def detectEyeCenters(face):
	eye_cascade = cv2.CascadeClassifier('src/haarcascade_eye.xml')
	eye_cascade_2 = cv2.CascadeClassifier('src/haarcascade_eye_2.xml')

	eyes = eye_cascade.detectMultiScale(face)
	
	if(len(eyes)<2):
		eyes = eye_cascade_2.detectMultiScale(face)

	print eyes	
	
	e1x = eyes[0][0]+(eyes[0][2]/2)
	e1y = eyes[0][1]+(eyes[0][3]/2)
	e2x = eyes[1][0]+(eyes[1][2]/2)
	e2y = eyes[1][1]+(eyes[1][3]/2)
	
	if (e1x>e2x) :
		left_eye_center_x = e2x
		left_eye_center_y = e2y
		right_eye_center_x = e1x
		right_eye_center_y = e1y
	else :
		left_eye_center_x = e1x
		left_eye_center_y = e1y
		right_eye_center_x = e2x
		right_eye_center_y = e2y

	return left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y
