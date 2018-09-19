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

import csv

data = []
with open('/Users/eeshannarula/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv', newline='') as csvfile:
    file = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in file:
        data.append(row)
        
del data[489]
del data[753]
del data[935]
del data[1080]
del data[1337]
del data[3327]
del data[3821]
del data[4374]
del data[5211]
del data[6662]
del data[6745]
del data[0]

def runForThree(var,cats):
        if var == cats[0]:
            return 0
        elif var == cats[1]:
             return 1
        elif var == cats[2]:
             return  2
        else :
             return var
def runForFour(var,cats):
        if var == cats[0]:
            return 0
        elif var == cats[1]:
             return 1
        elif var == cats[2]:
             return  2
        elif var == cats[3]:
             return  3    
        else :
             return var
              
def runForTwo(cat,cats):
        if cat == cats[0]:
            return 0
        elif cat == cats[1]:
            return 1
        else:
           return cat
       

targets = []
for row in data:
    del row[0]
    ## 0 if no 1 if yes
    subarray = [0] * 2
    if row[len(row)-1] == 'No':
        subarray[0] = 1
    else: subarray[1] = 1
    targets.append(subarray)
count = 0
for row in data:
    count +=1
    del row[len(row)-1]
    row[0] = runForTwo(row[0],['Male','Female'])
    row[2] = runForTwo(row[2],['No','Yes'])
    row[3] = runForTwo(row[3],['No','Yes'])
    row[5] = runForTwo(row[5],['No','Yes'])
    row[6] = runForThree(row[6],['No','Yes','No phone service'])
    row[7] = runForThree(row[7],['DSL','Fiber optic','No'])
    row[8] = runForThree(row[8],['No','Yes','No internet service'])
    row[9] = runForThree(row[9],['No','Yes','No internet service'])
    row[10] = runForThree(row[10],['No','Yes','No internet service'])
    row[11] = runForThree(row[11],['No','Yes','No internet service'])
    row[12] = runForThree(row[12],['No','Yes','No internet service'])
    row[13] = runForThree(row[13],['No','Yes','No internet service'])
    row[14] = runForThree(row[13],['Month-to-month','One year','Two year'])
    row[15] = runForTwo(row[15],['No','Yes'])
    row[16] = runForFour(row[16],['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'])
    if row[1] != 'SeniorCitizen':
        row[1] = int(row[1])
    if row[4] != 'tenure':
         row[4] = int(row[4])/72   
    if row[17] != 'MonthlyCharges':
         row[17] = round(float(row[17]))/119
    if row[18] != 'TotalCharges':
         row[18] =float(row[18])/8680
         
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(19,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(np.array(data),np.array(targets),batch_size=100, epochs=50,shuffle=True)
