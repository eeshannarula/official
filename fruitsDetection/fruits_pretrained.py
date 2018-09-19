import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import model_from_json

import cv2
import os

# load json and create model
json_file = open('/Users/eeshannarula/Documents/py/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/Users/eeshannarula/Documents/py/model.h5")
print("Loaded model from disk")

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd)

import numpy as np
import cv2

def maxnum(array):
    m = 0
    for i in array:
        if i > m:
            m = i
    return list(array).index(m)

def check(prediction):
    if prediction == 0:
        return 'apple_braeburn'
    elif prediction == 1:
        return 'apple_golden'
    elif prediction == 2:
        return 'apple_red'
    elif prediction == 3:
        return 'apricot'
    elif prediction == 4:
        return 'banana'
    elif prediction == 5:
        return 'cherry'
    elif prediction == 6:
        return 'dates'
    elif prediction == 7:
        return 'lemon'
    elif prediction == 8:
        return 'mango'
    elif prediction == 9:
        return 'pear'

img = cv2.imread('/Users/eeshannarula/Documents/fruitsDetction/Training/Dates/r_327_100.jpg',3)
array = []
for i in img:
        for j in i:
            array.append(( j[0] * 0.3+ j[1] * 0.59+0.11* j[2]) / 255)
ys = np.array(array).reshape(1,100,100,1)
print(check(maxnum(loaded_model.predict(ys)[0])))

cam = cv2.VideoCapture(1)
def click():
    s, im = cam.read() # captures image
    cv2.imshow("Test Picture", im) # displays captured image
    for i in im:
        for j in i:
            array.append(( j[0] * 0.3+ j[1] * 0.59+0.11* j[2]) / 255)
    ys = np.array(array).reshape(1,100,100,1)
    print(check(maxnum(loaded_model.predict(ys)[0])))
