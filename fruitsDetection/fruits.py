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

#making a sequential model
model = Sequential()

#adding layers to the model


model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#function to load images

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(list(img))
    return images

class Fruit:
    def __init__(self,data,label):
        self.data = data
        self.label = label
        self.inputs = None
        self.labels = None 
        
    def addInputs(self,inputs):
        imgs = len(inputs[0])
        self.inputs = np.array(inputs[0]).reshape(imgs,100,100,1)
        self.labels = np.array(inputs[1]).reshape(imgs,10) 
     
  
#convert to grey scaled
def getinputs(ar):
   a = ar.data
   l = ar.label
   array = []
   labels = []
   for img in a:
       subarray = []
       label = [0,0,0,0,0,0,0,0,0,0]
       label[l] = 1 
       for row in img:
           for block in row:
              r = block[0]
              g = block[1]
              b = block[2]
              subarray.append((r * 0.3 + g * 0.59 + b * 0.11)/255)
       array.append(subarray)
       labels.append(label)
   return [array,labels]

  
## loading images
print('loading Data (---------------------)')
apple_braeburn = Fruit(load_images_from_folder('/Users/eeshannarula/Documents/fruitsDetction/Training/Apple Braeburn'),0)
apple_braeburn.addInputs(getinputs(apple_braeburn))
print('loading Data (=>-------------------)')
apple_golden = Fruit(load_images_from_folder('/Users/eeshannarula/Documents/fruitsDetction/Training/Apple Golden 1'),1)
apple_golden.addInputs(getinputs(apple_golden))
print('loading Data (==>------------------)')
apple_red = Fruit(load_images_from_folder('/Users/eeshannarula/Documents/fruitsDetction/Training/Apple Red 1'),2)
apple_red.addInputs(getinputs(apple_red))
print('loading Data (===>-----------------)')
apricot = Fruit(load_images_from_folder('/Users/eeshannarula/Documents/fruitsDetction/Training/Apricot'),3)
apricot.addInputs(getinputs(apricot))
print('loading Data (====>----------------)')
banana = Fruit(load_images_from_folder('/Users/eeshannarula/Documents/fruitsDetction/Training/Banana'),4)
banana.addInputs(getinputs(banana))
print('loading Data (======>--------------)')
cherry = Fruit(load_images_from_folder('/Users/eeshannarula/Documents/fruitsDetction/Training/Cherry 1'),5)
cherry.addInputs(getinputs(cherry))
print('loading Data (=======>-------------)')
dates  = Fruit(load_images_from_folder('/Users/eeshannarula/Documents/fruitsDetction/Training/Dates'),6)
dates.addInputs(getinputs(dates))
print('loading Data (========>------------)')
lemon = Fruit(load_images_from_folder('/Users/eeshannarula/Documents/fruitsDetction/Training/Lemon'),7)
lemon.addInputs(getinputs(lemon))
print('loading Data (==========>----------)')
mango = Fruit(load_images_from_folder('/Users/eeshannarula/Documents/fruitsDetction/Training/Mango'),8)
mango.addInputs(getinputs(mango))
print('loading Data (=============>------)')
pear = Fruit(load_images_from_folder('/Users/eeshannarula/Documents/fruitsDetction/Training/Pear'),9)
pear.addInputs(getinputs(pear))
print('loading Data (==================>)')


def makeone(a):
        array = []
        for i in a:
            array.append(list(i))
        return np.array(array)


def concat(array):
    result  = []
    labels = []
    for a in array:
        for i  in a.inputs:
            result.append(i.copy())
        for i in a.labels:
            labels.append(i.copy())
    return [makeone(result),makeone(labels)]


friuts = [apple_braeburn,apple_golden,apple_red,apricot,banana,cherry,dates,lemon,mango,pear]

def calcTotalLen():
    sumnum = 0
    for i  in fruits:
        sumnum += len(i.inputs)
    return sumnum

x_train,y_train = concat(friuts)

model.fit(x_train, y_train, batch_size=100, epochs=50,shuffle=True)

def maxnum(array):
    m = 0
    for i in array:
        if i > m:
            m = i
    return list(array).index(m)

def test():
        score = 0
        total = calcTotalLen()
        imgs = x_train.copy()
        predictionsarray = model.predict(imgs)
        predictions = []
        correct = []
        for i in predictionsarray:
            predictions.append(maxnum(i))
        for i in y_train:
            correct.append(maxnum(i))

        for i in range(total):
            c = correct[i]
            p = predictions[i]
            if c == p :
                score+=1

        return (score/total) * 100        
        



