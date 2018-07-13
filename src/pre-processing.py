import detection as dtcn
import matplotlib.pyplot as plt
import cv2

def process(image):	
	model = "src/deploy.prototxt.txt"							# model-definition
	weights = "src/res10_300x300_ssd_iter_140000.caffemodel"	# pre-trained weights
	image = image												# images are loaded as 3D matrix - (h x w x c)
	
	colorImage, grayImage, faceBox = dtcn.detect(model,weights,image)
	# cropping the main face out of the GRAY SPACE image
	# as LBPH work on gray scaled images
	face = grayImage[faceBox[2]:faceBox[3], faceBox[0]:faceBox[1] ]
	return face

