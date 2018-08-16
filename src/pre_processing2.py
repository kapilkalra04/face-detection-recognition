import detection
import matplotlib.pyplot as plt
import cv2
import alignment

def detectMainFace(imageName,isPath):	
	model = "src/deploy.prototxt.txt"							# model-definition
	weights = "src/res10_300x300_ssd_iter_140000.caffemodel"	# pre-trained weights
	image = imageName											# image name reqd. images are loaded as 3D matrix - (h x w x c)	

	# send for face detection
	colorImage, grayImage, mainFaceBox = detection.detect(model,weights,image,isPath)
	
	# crop the misaligned face from the whole image
	mainFaceGray = grayImage[mainFaceBox[2]:mainFaceBox[3], mainFaceBox[0]:mainFaceBox[1]]
	mainFaceColor = colorImage[mainFaceBox[2]:mainFaceBox[3], mainFaceBox[0]:mainFaceBox[1]]

	return colorImage, mainFaceColor, mainFaceGray, mainFaceBox

def alignImage(colorImage,mainFaceGray,mainFaceBox):
	# obtain eye centers
	left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y = alignment.detectEyeCenters(mainFaceGray)
	
	# obtain affine transformation values
	center, angle, scale = alignment.rotate(mainFaceGray,left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y)
	
	# update co-ordinates according to colorImage the orignal iage
	left_eye_center_x = left_eye_center_x + mainFaceBox[0]
	right_eye_center_x = right_eye_center_x + mainFaceBox[0]
	left_eye_center_y = left_eye_center_y + mainFaceBox[2]
	right_eye_center_y = right_eye_center_y + mainFaceBox[2]
	center = (center[0]+mainFaceBox[0],center[1]+mainFaceBox[2])

	# perform affine transformation
	M = cv2.getRotationMatrix2D(center, angle, scale)
	alignedImage = cv2.warpAffine(colorImage,M,(colorImage.shape[1],colorImage.shape[0]),flags=cv2.INTER_CUBIC)
	
	return alignedImage, left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y

# return the face in gray scale
def getFaceGray(imagePath):
	# detect the misaligned largest face in gray
	colorImage, mainFaceColor, mainFaceGray, mainFaceBox = detectMainFace(imagePath,True)
	
	# straighten the actual image
	alignedImage, e1x, e1y, e2x, e2y = alignImage(colorImage,mainFaceGray,mainFaceBox)
	
	# detect the aligned largest face in gray
	colorImage, mainFaceColor, mainFaceGray, mainFaceBox = detectMainFace(alignedImage,False)
	
	# apply denoising
	mainFaceGray = cv2.fastNlMeansDenoising(mainFaceGray)										# denoising
		
	return mainFaceGray																			# returns a grayscaled,aligned,(256,256) face

# return the face in RGB
def getFaceColor(imagePath):
	# detect the misaligned largest face in gray
	colorImage, mainFaceColor, mainFaceGray, mainFaceBox = detectMainFace(imagePath,True)
	
	# straighten the actual image
	alignedImage, e1x, e1y, e2x, e2y = alignImage(colorImage,mainFaceGray,mainFaceBox)
	
	# detect the aligned largest face in gray
	colorImage, mainFaceColor, mainFaceGray, mainFaceBox = detectMainFace(alignedImage,False)
	
	# apply denoising
	mainFaceColor = cv2.fastNlMeansDenoisingColored(mainFaceColor)								# denoising
	
	return mainFaceColor																		# returns a grayscaled,aligned,(256,256) face

if __name__ == '__main__':
	
	plt.subplot(2,2,1)
	colorImage, mainFaceColor, mainFaceGray, mainFaceBox = detectMainFace('data/library/train/1.jpg',True)
	plt.imshow(colorImage)

	plt.subplot(2,2,2)
	plt.imshow(mainFaceColor)
	
	alignedImage, e1x, e1y, e2x, e2y = alignImage(colorImage,mainFaceGray,mainFaceBox)
	X = [e1x,e2x]
	Y = [e1y,e2y]
	plt.subplot(2,2,3)
	plt.imshow(alignedImage)
	plt.plot(X,Y,'-D',markersize=3)

	plt.subplot(2,2,4)
	# plt.imshow(alignedImage,cmap='gray')
	# plt.show()
	
	colorImage, mainFaceColor, mainFaceGray, mainFaceBox = detectMainFace(alignedImage,False)
	plt.imshow(mainFaceColor)
	plt.show()
	

	# plt.imshow(getFace('data/library/train/IMG_0007.JPG'),cmap='gray')
	# plt.show()
