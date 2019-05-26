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
data_dir='/data/train/images'
path='/data/train/annotations'
files = os.listdir(os.getcwd()+path)

x_train=[]
y_train=[]
for img in tqdm(os.listdir(os.getcwd()+data_dir)):
	try:
 		path1 = os.path.join(os.getcwd()+data_dir,img)
 		print(path1)
 		img = cv2.imread(path1,cv2.IMREAD_COLOR)
 		img=cv2.resize(img,(150,150))
 		height, width, channels = img.shape
 		print (height, width, channels)
 		x_train.append(np.array(img))
	except Exception as e:
 		print(str(e))
print(len(x_train))

os.chdir(os.getcwd()+path)
for filename in tqdm(files):
	a=np.ndarray(shape=(4), dtype=int, order='C')
	f = open(filename, "r")
	count=0
	text = f.readlines()
	for rd in text:
		count+=1
		if count==19:
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
	y_train.append(a)
print(len(y_train))

os.chdir('/home/yash/auv/keras')
print(os.getcwd())
data_dir='/data/test/images'
path='/data/test/annotations'
files = os.listdir(os.getcwd()+path)
x_test=[]
y_test=[]
for img in tqdm(os.listdir(os.getcwd()+data_dir)):
	try:
 		path1 = os.path.join(os.getcwd()+data_dir,img)
 		print(path1)
 		img = cv2.imread(path1,cv2.IMREAD_COLOR)
 		img=cv2.resize(img,(150,150))
 		height, width, channels = img.shape
 		print (height, width, channels)
 		x_test.append(np.array(img))
	except Exception as e:
 		print(str(e))
print(len(x_test))

os.chdir(os.getcwd()+path)
for filename in tqdm(files):
	a=np.ndarray(shape=(4), dtype=int, order='C')
	f = open(filename, "r")
	count=0
	text = f.readlines()
	for rd in text:
		count+=1
		if count==19:
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
	y_test.append(a)
print(len(y_test))

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

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

batch_size = 16
epochs = 20


opt = optimizers.Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# train the model
print("[INFO] training model...")
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# json_string = model.to_json()
# open('localizer.json', 'w').write(json_string)
# yaml_string = model.to_yaml()
# open('localizer.yaml', 'w').write(yaml_string)

# # save the weights in h5 format
# model.save_weights('localizer_wts.h5')

# # to read a saved model and weights
# # model = model_from_json(open('my_model_architecture.json').read())
# # model = model_from_yaml(open('my_model_architecture.yaml').read())
# # model.load_weights('my_model_weights.h5')




pyplot.plot(history.history['mean_squared_error'])
pyplot.show()