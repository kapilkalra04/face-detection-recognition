import cv2
import numpy as np
from keras.models import load_model
 
def align(face,left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y):
	lx = left_eye_center_x
	ly = left_eye_center_y
	rx = right_eye_center_x
	ry = right_eye_center_y

	desiredHeight = 256
	desiredWidth = 256
	desiredLeftEye = 64
	desiredRightEye = 192

	dY = ry - ly
	dX = rx - lx

	angle = np.degrees(np.arctan2(dY, dX))				# angle should be in degrees
	
	eyeActlDist = np.sqrt((dY**2) + (dX**2))
	eyeDsrdDist = abs(desiredLeftEye - desiredRightEye)
	scale = eyeDsrdDist/eyeActlDist						# scale = zoom-in/zoom-out
	# scale = 1

	cx = (rx+lx)//2
	cy = (ry+ly)//2
	center = (cx,cy)
	
	M = cv2.getRotationMatrix2D(center, angle, scale)
	
	M[0,2] = M[0,2] + (desiredWidth/2 - cx)							# translation for center in x
	M[1,2] = M[1,2] + (desiredLeftEye - cy)							# translation for center in y 

	alignedFace = cv2.warpAffine(face,M,(256,256),flags=cv2.INTER_CUBIC)	# face.shape = (96,96)
	
	return alignedFace

# def computeTransformation(face):
# 	scaledFace = cv2.resize(face,(96,96))
# 	plt.imshow(scaledFace,cmap='gray')
# 	scaledFace = cv2.fastNlMeansDenoising(scaledFace)
# 	scaledFaceTemp = scaledFace[np.newaxis,:,:,np.newaxis]
# 	scaledFaceTemp = scaledFaceTemp/255
	
# 	print "[INF0] Loading the CNN"
# 	cnn2 = load_model('src/cnn62_1.h5')

# 	print "[INF0] Predicting Facial-Keypoints"
# 	results = cnn2.predict(scaledFaceTemp, batch_size=1)
# 	results = (results*48) + 48
# 	results = results.reshape(30)
# 	print results
	
# 	return results
	
# 	left_eye_center_x = results[0]
# 	left_eye_center_y = results[1]
# 	right_eye_center_x = results[2]
# 	right_eye_center_y = results[3]

# 	X = []
# 	Y = []

# 	X.extend([results[0],results[2],results[4],results[6]])
# 	Y.extend([results[1],results[3],results[5],results[7]])
# 	plt.plot(X,Y,'c*',markersize=3)
	
# 	scaledFace = np.squeeze(scaledFace,axis=0)
# 	scaledFace = np.squeeze(scaledFace,axis=2)
# 	alignedFace = align(scaledFace,left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y)
# 	plt.subplot(2,2,4)
# 	plt.imshow(alignedFace,cmap='gray')


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
