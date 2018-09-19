import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import model_from_json

import cv2
import os

model = Sequential()

model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.06, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

class catigory:
    def __init__(self,data,labels):
        self.data = data
        self.labels = labels

    @staticmethod
    def concat(a):
       array = []


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(list(img))
    return images

circles = np.array(load_images_from_folder('/Users/eeshannarula/Downloads/datas/shapes/circles'))
squares = np.array(load_images_from_folder('/Users/eeshannarula/Downloads/datas/shapes/squares'))
triangles = np.array(load_images_from_folder('/Users/eeshannarula/Downloads/datas/shapes/triangles'))

def getinputs(a,l):
   array = []
   labels = []
   for img in a:
       subarray = []
       label = [0,0,0]
       label[l] = 1 
       for row in img:
           for block in row:
              num = block[0]
              subarray.append(num/255)
       array.append(subarray)
       labels.append(label)
   return catigory(array,labels)

circles_cat = getinputs(circles,0)
squares_cat = getinputs(squares,1)
triangles_cat = getinputs(triangles,2)

circles_inp = np.array(circles_cat.data).reshape((100,28,28))
squares_inp = np.array(squares_cat.data).reshape((100,28,28))
triangels_inp = np.array(triangles_cat.data).reshape((100,28,28))

circles_lab = np.array(circles_cat.labels)
squares_lab = np.array(squares_cat.labels)
triangels_lab = np.array(triangles_cat.labels)

conv_circles = np.divide(circles_inp,255)
conv_squares= np.divide(squares,255)
conv_triangles= np.divide(triangles,255)

def concat(a,b,c):
    result  = []
    for i in a:
        result.append(i.copy())
    for i in b:
        result.append(i.copy()) 
    for i in c:
        result.append(i.copy())
    return result

def makeone(a):
        array = []
        for i in a:
            array.append(list(i))
        return np.array(array)


x_train = makeone(concat(circles_inp,squares_inp,triangels_inp)).reshape(300,28,28,1)
y_train = makeone(concat(circles_lab,squares_lab,triangels_lab))

            
##model.fit(x_train, y_train, batch_size=100, epochs=50,shuffle=True)
##keras.callbacks.TerminateOnNaN()

def maxnum(array):
    m = 0
    for i in array:
        if i > m:
            m = i
    return list(array).index(m)


# load json and create model
json_file = open('/Users/eeshannarula/Documents/py/shapes_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/Users/eeshannarula/Documents/py/shapes_model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
sgd = SGD(lr=0.06, decay=1e-6, momentum=0.9, nesterov=True)
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd)

def test():
        score = 0
        imgs = np.array(x_train).reshape(300,28,28,1)
        predictionsarray = loaded_model.predict(imgs)
        predictions = []
        correct = []
        for i in predictionsarray:
            predictions.append(maxnum(i))
        for i in y_train:
            correct.append(maxnum(i))

        for i in range(300):
            c = correct[i]
            p = predictions[i]
            if c == p :
                score+=1

        return (score/300) * 100        
        

        
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *

width = 500
height = 500
center = height//2
white = (255, 255, 255)
green = (0,128,0)

def save():
    filename = "image.png"
    image1.save(filename)

def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    draw.line([x1, y1, x2, y2],fill="black",width=5)

root = Tk()

# Tkinter create a canvas to draw on
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()

# PIL create an empty image and draw object to draw on
# memory only, not visible
image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

# do the Tkinter canvas drawings (visible)
# cv.create_line([0, center, width, center], fill='green')

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

# do the PIL image/draw (in memory) drawings
# draw.line([0, center, width, center], green)

# PIL image can be saved as .png .jpg .gif or .bmp file (among others)
# filename = "my_drawing.png"
# image1.save(filename)
button=Button(text="save",command=save)
button.pack()
root.mainloop()

def predict(img):
    image = cv2.imread(img', cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, (28, 28))
    array = []
    for i in resized_image:
          for j in i:
            array.append(j[0]/255)
    ys = np.array(array).reshape(1,28,28,1)
    return maxnum(loaded_model.predict(ys))