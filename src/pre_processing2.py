import detection
import matplotlib.pyplot as plt
import cv2
import alignment

def detectMainFace(imageName,isPath):	
	model = "src/deploy.prototxt.txt"							# model-definition
	weights = "src/res10_300x300_ssd_iter_140000.caffemodel"	# pre-trained weights
	image = imageName											# image name reqd. images are loaded as 3D matrix - (h x w x c)	

	colorImage, grayImage, mainFaceBox = detection.detect(model,weights,image,isPath)
	
	mainFaceGray = grayImage[mainFaceBox[2]:mainFaceBox[3], mainFaceBox[0]:mainFaceBox[1] ]
	
	return colorImage, mainFaceGray, mainFaceBox

def alignImage(colorImage,mainFaceGray,mainFaceBox):
	left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y = alignment.detectEyeCenters(mainFaceGray)
	
	center, angle, scale = alignment.rotate(mainFaceGray,left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y)
	
	X = [left_eye_center_x,right_eye_center_x]
	Y = [left_eye_center_y,right_eye_center_y]
	# plt.plot(X,Y,'-D',markersize=3)


	# update co-ordinates according to colorImage
	left_eye_center_x = left_eye_center_x + mainFaceBox[0]
	right_eye_center_x = right_eye_center_x + mainFaceBox[0]
	left_eye_center_y = left_eye_center_y + mainFaceBox[2]
	right_eye_center_y = right_eye_center_y + mainFaceBox[2]
	
	center = (center[0]+mainFaceBox[0],center[1]+mainFaceBox[2])

	M = cv2.getRotationMatrix2D(center, angle, scale)
	
	alignedImage = cv2.warpAffine(colorImage,M,(colorImage.shape[1],colorImage.shape[0]),flags=cv2.INTER_CUBIC)
	
	return alignedImage, left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y


def getFace(imagePath):
	colorImage, mainFaceGray, mainFaceBox = detectMainFace(imagePath,True)
	
	alignedImage, e1x, e1y, e2x, e2y = alignImage(colorImage,mainFaceGray,mainFaceBox)
	
	colorImage, mainFaceGray, mainFaceBox = detectMainFace(alignedImage,False)
	
	mainFaceGray = cv2.fastNlMeansDenoising(mainFaceGray)									# denoising
	
	return mainFaceGray																		# returns a grayscaled,aligned,(256,256) face

if __name__ == '__main__':
	
	plt.subplot(2,2,1)
	colorImage, mainFaceGray, mainFaceBox = detectMainFace('data/images/test9.JPG',True)
	plt.imshow(colorImage)

	plt.subplot(2,2,2)
	plt.imshow(mainFaceGray,cmap='gray')
	
	alignedImage, e1x, e1y, e2x, e2y = alignImage(colorImage,mainFaceGray,mainFaceBox)
	X = [e1x,e2x]
	Y = [e1y,e2y]
	plt.subplot(2,2,3)
	plt.imshow(alignedImage,cmap='gray')
	plt.plot(X,Y,'-D',markersize=3)

	plt.subplot(2,2,4)
	# plt.imshow(alignedImage,cmap='gray')
	# plt.show()
	
	colorImage, mainFaceGray, mainFaceBox = detectMainFace(alignedImage,False)
	print mainFaceGray.shape
	plt.imshow(mainFaceGray,cmap='gray')
	plt.show()
	

	# plt.imshow(getFace('data/library/train/IMG_0007.JPG'),cmap='gray')
	# plt.show()
