import numpy as np
import matplotlib.pyplot as plt
import cv2

def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def convertToGRAY(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def detect(model,weights,image,isPath):

	# specify locations of the 	model and its weights
	args = {}
	args["model"] = model					# model-definition
	args["weights"] = weights				# pre-trained weights
	args["image"] = image					# images are loaded as 3D matrix - (h x w x c)
	args["confidence"] = 0.75				# when confidence>value then it is a face

	# load the caffe model 
	print "[INFO] Loading model"
	# net = cnn used to detect faces
	net = cv2.dnn.readNetFromCaffe(args["model"], args["weights"])
	 
	# load the input image
	if(isPath==True): 
		image = cv2.imread(args["image"])
	else:
		image = image

	# print len(image)							# height of the image
	# print len(image[0])						# width of the image
	# print len(image[0][0])					# no of color-channels 
	# print image.shape							# stores h,w,c values
	(h, w) = image.shape[:2]

	# construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it 
	# along with doing a mean subtraction
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
	 1.0, (300, 300), (104.0, 177.0, 123.0))

	print "[INFO] Computing face detections..."
	net.setInput(blob)
	detections = net.forward()

	count = 0									# count of no of faces detected
	faces = {}									# stores the faces rectangles co-ordinates
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]
		
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for the
			# face
			faces[i] = []
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")	# extracting integral values
			# adding area details along with co-ordinate values
			faces[i].extend([startX,endX,startY,endY,((endX-startX)*(endY-startY))])

			# plotting the face rectangle	
			x = []
			y = []

			# plot the box
			x.extend([startX,endX,endX,startX,startX])
			y.extend([startY,startY,endY,endY,startY])
			# plt.plot(x,y)
			count = count + 1

	print "Faces Detected = " + str(count)
	
	largestFaceIndex = -1
	largestAreaYet = 0
	
	for i in range(0,len(faces)):
		if(faces[i][4]>largestAreaYet):
			largestFaceIndex = i
			largestAreaYet = faces[i][4]

	return convertToRGB(image),convertToGRAY(image),faces[largestFaceIndex]