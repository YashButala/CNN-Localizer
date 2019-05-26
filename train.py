import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D as Conv2D 
from keras.layers.convolutional import MaxPooling2D 
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import losses,optimizers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import tensorflow as tf
from PIL import Image
from numpy import *

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import re
from tqdm import tqdm
import os

import array as arr
a = arr.array('d', [1.1, 3.5, 4.5])
img_dir='/data/images'
ann_dir='/data/annotations'
files = os.listdir(os.getcwd()+ann_dir)
imgsize=150

def training_data():
	Y=[]
	X=[]
	for img in tqdm(os.listdir(os.getcwd()+img_dir)):
		path = os.path.join(os.getcwd()+img_dir,img)
		img = cv2.imread(path,cv2.IMREAD_COLOR)
		img = cv2.resize(img,(imgsize,imgsize))   
		X.append(np.array(img))
	print(len(X))
	os.chdir(os.getcwd()+ann_dir)
	for filename in tqdm(files):
		a=np.ndarray(shape=(4), dtype=int, order='C')
		f = open(filename, "r")
		count=0
		text = f.readlines()
		for rd in text:
			count+=1
			if count==19 :
				r = re.findall(r'\d+',rd) ##xmin
				r=map(int,r)
				a[0]=max(r)
			if count==20:	
				r = re.findall(r'\d+',rd) ##ymin
				r=map(int,r)
				a[1]=max(r)
			if count==21:	
				r = re.findall(r'\d+',rd) ##xmax
				r=map(int,r)
				a[2]=max(r)
			if count==22:	
				r = re.findall(r'\d+',rd) ##ymax
				r=map(int,r)
				a[3]=max(r)
		Y.append(a)
	print(len(Y))
	X=np.array(X)
	Y=np.array(Y)
	return X,Y

A,B=training_data()

x_train,x_test,y_train,y_test=train_test_split(A,B,random_state=36)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='relu',input_shape=x_train[0].shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())					
model.add(Dense(1000, activation='relu'))
model.add(Dense(400, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(4, activation='linear'))

batch_size = 100
epochs = 1


opt = optimizers.Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# train the model
print("[INFO] training model...")
history=model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

os.chdir(os.getcwd())
json_string = model.to_json()
open('localizer.json', 'w').write(json_string)
yaml_string = model.to_yaml()
open('localizer.yaml', 'w').write(yaml_string)

# # save the weights in h5 format
# model.save_weights('localizer_wts.h5')

# # to read a saved model and weights
# # model = model_from_json(open('my_model_architecture.json').read())
# # model = model_from_yaml(open('my_model_architecture.yaml').read())
# # model.load_weights('my_model_weights.h5')



plt.plot(history.history['mse'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

