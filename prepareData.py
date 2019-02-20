from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
import pandas as pd
import numpy as np
import random
import h5py
import os
import cv2

CSVpath = "train_labels.csv"
Dst = "D:/DeepLearning/histopathologic/Dataset/"
Src = "D:/DeepLearning/histopathologic/train/"

def segregateData(path):
	data = pd.read_csv(path)
	for i in range(0,len(data["id"])):
		if data["label"][i] == 0:
			trainDst = Dst + "no_cancer/"  + str(data["id"][i]) + ".tif"
			trainSrc = Src + str(data["id"][i]) + ".tif"
			os.rename(trainSrc,trainDst)
		else:
			trainDst = Dst + "cancer/"  + str(data["id"][i]) + ".tif"
			trainSrc = Src + str(data["id"][i]) + ".tif"
			os.rename(trainSrc,trainDst)

def list_files(basePath, validExts=None, contains=None):
	for (rootDir, dirNames, filenames) in os.walk(basePath):
		for filename in filenames:
			if contains is not None and filename.find(contains) == -1:
				continue
			ext = filename[filename.rfind("."):].lower()
			if validExts is None or ext.endswith(validExts):
				imagePath = os.path.join(rootDir, filename)
				yield imagePath

def splitData(inputPath):

	testPath =  "D:/DeepLearning/histopathologic/Dataset/test/cancer/"
	trainPath = "D:/DeepLearning/histopathologic/Dataset/train/cancer/"
	validPath = "D:/DeepLearning/histopathologic/Dataset/validation/cancer/"
	imagePaths = sorted((list(list_files(inputPath,validExts=".tif",contains=None))))
	trainSize = int(0.8 * len(imagePaths))
	testSize = int(len(imagePaths) - trainSize)
	# trainImages = imagePaths [trainSize:]
	# testImages = imagePaths [:-testSize]
	validImages = imagePaths[testSize:]
	datasets = [("validation",validImages,validPath)]#("Test",testImages,testPath),("Train",trainImages,trainPath)]
	for splitType,imagePaths,outputPath in datasets:
		for n,paths in enumerate(imagePaths):
			print(splitType+": "+str(n)+"/"+str(len(imagePaths)))
			src = paths
			fileDst = outputPath + paths.split("/")[-1]
			os.rename(src,fileDst)
