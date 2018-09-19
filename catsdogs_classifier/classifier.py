## Author ~ Eeshan Narula
## this is a code for making a cats-dogs image classifier
## we would be using keras machine learning lib. 
import numpy as np 
import keras as ks 

## extracting out model and layers from keras
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import SGD

## for image rendering
import cv2
import os

## function to load the images

def importImages(folder):
    images = []
    for image in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,image))
        if img is not None:
            images.append(list(cv2.resize(img,(100,100))))
    return np.array(images)


## now we will import the images
cats_data = importImages('catsdogs/training_set/cats')
dogs_data = importImages('catsdogs/training_set/dogs')
print('Done')

## class for an animal

class animal:
    def __init__(self,data,label):
        self.data = data
        self.label = label
        self.imgs = len(data)
        self.targets = self.makeLabels()

    def makeLabels(self):
        singleTarget = [0,0]
        singleTarget[self.label] = 1
        targetList = [singleTarget] * self.imgs
        return np.array(targetList)


 
cats = animal(cats_data,0)
dogs = animal(dogs_data,1)

## concat the dogs and cats data to make TrainingData
def concat():
    xs = []
    ys = []
    for img in cats.data:
        xs.append(img)
    for img in dogs.data:
        xs.append(img)    
    for label in cats.targets:
        ys.append(label) 
    for label in  dogs.targets:
        ys.append(label)
    return np.array(xs)/255,np.array(ys)

xs,ys = concat()

## Making the model

model = Sequential()

model.add(Conv2D(kernel_size = (3,3),filters = 8,activation = 'relu',input_shape = [100,100,3]))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(kernel_size = (3,3),filters = 16,activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(kernel_size = (3,3),filters = 32,activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(100))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(2,activation = 'sigmoid'))

optimizer = SGD(lr = 0.01,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer,loss = 'categorical_crossentropy')

## training the model
model.fit(xs,ys,epochs=1,shuffle=True)

## testing the model
cats_testing = importImages('catsdogs/test_set/cats')
testimg = np.divide(np.array(cats_testing[0]),255)
prediction = model.predict(np.divide(np.array(cats_testing),255))
print(prediction) 