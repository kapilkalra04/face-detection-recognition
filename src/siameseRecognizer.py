import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import pre_processing2 as pre

empEmbeddings = np.load('src/empEmbeddings.npy')
print empEmbeddings.shape

cstmrEmbeddings = np.load('src/cstmrEmbeddings.npy')
print cstmrEmbeddings.shape

faceListTrain = []
faceListTest = []

for imagePath in glob.glob('data/library/train2/*'):
	faceListTrain.append(cv2.resize(pre.getFaceColor(imagePath),(160,160)))

for imagePath in glob.glob('data/library/test2/*'):
	faceListTest.append(cv2.resize(pre.getFaceColor(imagePath),(160,160)))

plt.subplot2grid((1,4),(0,0))
plt.imshow(faceListTrain[0])

for i in range(0,len(empEmbeddings)):
	for j in range(0,len(cstmrEmbeddings)):
		plt.subplot2grid((1,4),(0,j+1))
		plt.imshow(faceListTest[j])
		plt.title(np.linalg.norm(empEmbeddings[i] - cstmrEmbeddings[j]))



plt.tight_layout()
plt.suptitle('ONE SHOT LEARNING TEST')
plt.show()
