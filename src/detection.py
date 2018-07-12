import numpy as np
import matplotlib.pyplot as plt
import cv2

def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# specify locations of the 	model and its weights
args = {}
args["prototxt"] = "src/deploy.prototxt.txt"					# model-definition
args["model"] = "src/res10_300x300_ssd_iter_140000.caffemodel"	# pre-trained weights
args["image"] = "src/images/test6.png"							# images are loaded as 3D matrix - (h x w x c)
args["confidence"] = 0.75										# confidence>value is a face

# load the caffe model 
print "[INFO] Loading model"
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
 
# load the input image 
image = cv2.imread(args["image"])

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

for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]
	
	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")	# extracting integral values
		x = []
		y = []

		# plot the box
		x.extend([startX,endX,endX,startX,startX])
		y.extend([startY,startY,endY,endY,startY])
		plt.plot(x,y)
		
		print ">> DETECTION"

plt.imshow(convertToRGB(image))
plt.show() 
# show the output image
# cv2.imshow("Output", image)
# cv2.waitKey(0)
