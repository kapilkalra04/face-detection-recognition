import detection
import matplotlib.pyplot as plt
import cv2
import alignment

def detectMainFace(imageName):	
	model = "src/deploy.prototxt.txt"							# model-definition
	weights = "src/res10_300x300_ssd_iter_140000.caffemodel"	# pre-trained weights
	image = imageName											# image name reqd. images are loaded as 3D matrix - (h x w x c)	

	colorImage, grayImage, faceBox = detection.detect(model,weights,image)
	
	#[startX,endX,startY,endY,area] = faceBox
	
	# cropping the main face out of the GRAY SPACE image
	# as LBPH work on gray scaled images
	mainFaceGray = grayImage[faceBox[2]:faceBox[3], faceBox[0]:faceBox[1] ]
	return colorImage, mainFaceGray, faceBox

def alignMainFace(image):
	# scaleX = 96.0/(image.shape[0])						# ROWS
	# scaleY = 96.0/(image.shape[1])						# COLS

	#results = alignment.computeTransformation(image)
	# for i in range(0,len(results)):
	# 	if i % 2 :
	# 		results[i] = results[i]*(1/scaleX)
	# 	else :
	# 		results[i] = results[i]*(1/scaleY)
	
	# left_eye_center_x = results[0]
	# left_eye_center_y = results[1]
	# right_eye_center_x = results[2]
	# right_eye_center_y = results[3]

	# X = []
	# Y = []

	# X.extend([results[0:29:2]])
	# Y.extend([results[1:30:2]])
	
	# plt.plot(X,Y,'c*',markersize=3)

	left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y = alignment.detectEyeCenters(image)
	# alignedFace.shape = (256,256)
	alignedFace = alignment.align(image,left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y)
	return alignedFace, left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y
	

def getFace(image):
	colorImage, mainFaceGray, mainFaceBox = detectMainFace(image)
	
	alignedFace,left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y = alignMainFace(mainFaceGray)
	
	alignedFace = cv2.fastNlMeansDenoising(alignedFace)					# denoising
	
	return alignedFace													# returns a grayscaled,aligned,(256,256) face

if __name__ == '__main__':
	
	plt.subplot(2,2,1)
	colorImage, mainFaceGray, mainFaceBox = detectMainFace('data/images/test2.jpg')
	plt.imshow(colorImage)

	plt.subplot(2,2,2)
	plt.imshow(mainFaceGray,cmap='gray')
	
	plt.subplot(2,2,3)
	alignedFace, e1x, e1y, e2x, e2y = alignMainFace(mainFaceGray)
	X = [e1x,e2x]
	Y = [e1y,e2y]
	plt.imshow(mainFaceGray,cmap='gray')
	plt.plot(X,Y,'-D',markersize=3)

	plt.subplot(2,2,4)
	plt.imshow(alignedFace,cmap='gray')
	
	plt.show()
	
	getFace('data/images/test2.jpg')
