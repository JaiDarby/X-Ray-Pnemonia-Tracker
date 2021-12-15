import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import random
import pickle


DataDir = "Datasets/Xrays"
Categories = ["Normal" , "Pneumonia"]

for category in Categories:
    path = os.path.join(DataDir, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), 1)
        #plt.imshow(img_array, cmap='gray')  # graph it
        #plt.show()

ImgSize = 75

TrainingData = []

def TrainData():
    for category in Categories:
        path = os.path.join(DataDir, category)
        ClassNum = Categories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), 0)
            new_array = cv2.resize(img_array,(ImgSize,ImgSize))
            TrainingData.append([new_array,ClassNum])

TrainData()

random.shuffle(TrainingData)

x = []
y = []

for features, label in TrainingData:
    x.append(features)
    y.append(label)

print(len(x))
print(len(y))

x = np.array(x).reshape(-1, ImgSize, ImgSize, 1)
y= np.array(y)

print(len(x))
print(len(y))

pickle_out = open("DatasetX.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("DatasetY.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print("Dataset Created")