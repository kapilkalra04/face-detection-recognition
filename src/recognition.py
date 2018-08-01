import glob
import numpy as np
import pre_processing2 as pre
import cv2
import matplotlib.pyplot as plt

images = []
for imagePath in glob.glob('data/library/train/*'):
	images.append(imagePath)

faceList = []
labelList = [0,0,0,0,0,0,0,0,0,0]

index = 0

for path in images:
	temp = pre.getFace(path)
	temp = cv2.resize(temp,(369,512))
	faceList.append(temp)			
	print "[INFO] Image Loaded: " + str(index+1)
	print faceList[-1].shape
	
	plt.subplot2grid((5,3),(index%5,index/5))
	plt.imshow(faceList[-1])
	index = index + 1


print labelList
faceRecognizer = cv2.face.LBPHFaceRecognizer_create(1,8,8,8,123)
faceRecognizer.train(faceList,np.array(labelList))

imagesTest = []
for imagePath in glob.glob('data/library/test/*'):
	imagesTest.append(imagePath)

print "[INFO] ========TESTING======="
faceListTest = []
prediction = {}
index = 0
for path in imagesTest:
	testSample = pre.getFace(path)			#np.array.shape = (256,256)
	testSample = cv2.resize(testSample,(369,512))
	print "[INFO] Test Image Loaded: " + str(index+1)
	prediction[index] = []
	predictedLabel, confidence = faceRecognizer.predict(testSample)
	
	plt.subplot2grid((5,3),(index,2))
	plt.imshow(testSample,cmap='gray')
	plt.title(str(predictedLabel) + " : " + str(confidence))
	
	prediction[index].extend([predictedLabel,confidence])	
	index = index + 1
	

plt.tight_layout()
plt.show()
print prediction
