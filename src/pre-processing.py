import detection as dtcn
import matplotlib.pyplot as plt
import cv2

model = "src/deploy.prototxt.txt"							# model-definition
weights = "src/res10_300x300_ssd_iter_140000.caffemodel"	# pre-trained weights
image = "data/images/test2.jpg"							# images are loaded as 3D matrix - (h x w x c)
confidence = 0.75											# when confidence>value then it is a face

image, faceBox = dtcn.detect(model,weights,image,confidence)
print len(image)
face = image[faceBox[2]:faceBox[3], faceBox[0]:faceBox[1] ]
plt.imshow(image)
plt.subplot(2,1,2)
plt.imshow(face)
plt.show() 


