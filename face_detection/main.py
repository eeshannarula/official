# Author ~ Eeshan Narula
# dataset ~ https://www.kaggle.com/drgilermo/face-images-with-marked-landmark-points/home



import numpy as np 
import pandas as pd
import os

# Loading images and points
image = np.moveaxis(np.load('/Users/eeshan/Downloads/input/face_images.npz')['face_images'],-1,0)
points = pd.read_csv('/Users/eeshan/Downloads/input/facial_keypoints.csv')

# Taking only center points for easy training
selected_points=np.nonzero(points.left_eye_center_x.notna() & points.right_eye_center_x.notna() &
         points.nose_tip_x.notna() & points.mouth_center_bottom_lip_x.notna())[0]

# normalizing the data
normalization_val=image.shape[1]
m=selected_points.shape[0]
X=np.zeros((m,normalization_val,normalization_val,1))
Y=np.zeros((m,8))

X[:,:,:,0]=image[selected_points,:,:]/255.0  
Y[:,0]=points.left_eye_center_x[selected_points]/normalization_val
Y[:,1]=points.left_eye_center_y[selected_points]/normalization_val
Y[:,2]=points.right_eye_center_x[selected_points]/normalization_val
Y[:,3]=points.right_eye_center_y[selected_points]/normalization_val
Y[:,4]=points.nose_tip_x[selected_points]/normalization_val
Y[:,5]=points.nose_tip_y[selected_points]/normalization_val
Y[:,6]=points.mouth_center_bottom_lip_x[selected_points]/normalization_val
Y[:,7]=points.mouth_center_bottom_lip_y[selected_points]/normalization_val


# Split the dataset
from sklearn.model_selection import train_test_split

random_seed=21
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=random_seed)

#training the model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import SGD

#constructiong the model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding = 'same', activation='tanh', input_shape=(normalization_val, normalization_val, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='sigmoid'))

#optimizer
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)


# Appling fit function
model.fit(Xtrain, Ytrain, batch_size=128, epochs=10, validation_data = (Xtest, Ytest), verbose = 1)

# testing the model
Ytest_pred = np.array(model.predict(Xtest)) * normalization_val