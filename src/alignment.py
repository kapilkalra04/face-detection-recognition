import cv2

def align(face,left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y):
	lx = left_eye_center_x
	ly = left_eye_center_y
	rx = right_eye_center_x
	ry = right_eye_center_y

	desiredHeight = 96
	desiredWidth = 96
	desiredLeftEye = 63
	desiredRightEye = 33

	dY = ry - ly
	dX = rx - lx

	angle = np.arctan(dy/(abs(dx)))*180/np.pi			# angle should be in degrees
	
	eyeActlDist = np.sqrt((dY**2) + (dX**2))
	eyeDsrdDist = desiredLeftEye - desiredRightEye
	scale = eyeDsrdDist/eyeActlDist						# scale = zoom-in/zoom-out

	cx = (rx+lx)/2
	cy = (ry+ly)/2
	center = (cx,cy)
	
	M = cv2.getRotationMatrix2D(center, angle, scale)

	M[0,2] = M[0,2] + (48 - cx)							# translation for center in x
	M[1,2] = M[1,2] + (33 - cy)							# translation for center in y 

	alignedFace = cv2.warpAffine(face,M,(96,96),flags=cv2.INTER_CUBIC)	# face.shape = (96,96)
	return alignedFace


