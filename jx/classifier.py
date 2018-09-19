import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import model_from_json

import tkinter
import cv2
import os 

class Images:
    
    @staticmethod
    def loadImages(folder):
      images = []
      for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(list(img))
      return images

    @staticmethod
    def prepareInputs(imgs,label,totalClasses):
        array = []
        labels = []
        for img in imgs:
            subarray = []
            target = [0] * totalClasses
            target[label] = 1
            for row in img:
                for block in row:
                    r = block[0]
                    g = block[1]
                    b = block[2]
                    subarray.append((r * 0.3 + g * 0.59 + b * 0.11)/255)
            array.append(subarray)        
            labels.append(target) 
        return [array,labels] 
    @staticmethod
    def makeone(a):
        array = []
        for i in a:
            array.append(list(i))
        return np.array(array)

    @staticmethod
    def concat(array,lab):
      result  = []
      labels = []
      for a in array:
          for i  in a:
            result.append(i.copy())
      for a in lab:
          for i in a:
            labels.append(i.copy())
      return [Images.makeone(result),Images.makeone(labels)]


class Classifier:
    def __init__(self,classes):
        self.classes = classes
        self.data = []
        self.labels = []
        
    def addData(self,path,label):
        imgs = Images.loadImages(path)
        for img in imgs:
            rendered = np.array(img)
            cv2.resize(rendered, (100, 100)) 
        data = Images.prepareInputs(imgs,label,self.classes)
        self.data.append(data[0])
        self.labels.append(data[1])
        print('done')
    
    def compileData(self):
        combined = Images.concat(self.data,self.labels) 
        self.data = combined[0]
        self.labels = combined[1]
        
    def compileModel(self,learningRate):
        #making a sequential model
        self.model = Sequential()

        #adding layers to the model


        self.model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(100, 100, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

        sgd = SGD(lr=learningRate, decay=1e-6, momentum=0.9, nesterov=True)
        self. model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    def train(self,epochs,batchsize):
        self.model.fit(np.array(self.data),np.array(self.labels), batch_size=batchsize, epochs=epochs,shuffle=True) 

    
    def savemodel(self,name):
        model_json = self.model.to_json()
        with open(name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(name + ".h5")
        print("Saved model to disk")
        