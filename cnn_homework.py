#!/usr/bin/env python
# coding: utf-8

# In[26]:


from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
import csv
import os

from skimage.feature import daisy
from sklearn.ensemble import RandomForestClassifier

# # Write your own feature extractor
# def my_feature_extractor(image):
#     #return None
#     return daisy(image, step=20, radius=15, rings=3, histograms=6,orientations=8, visualize=False).flatten()
#
#
# # This function will run only if you dowloaded the data, put them in correct location
# # and your feature extractor is working
# # for testing purposes you can load small number of samples first using parameter n_samples
#
# def load_data_and_extract_features(n_samples):
#     # Load image names and training labels
#     try:
#         csv_reader = csv.reader(open('./train_labels_sample.csv', 'r'))
#     except Exception:
#         print("Could not load train_labels_sample.csv, are you sure you've downloaded it?")
#         raise
#     headers = next(csv_reader, None)  # Skips first line of csv
#
#     # Creating features and labels for training set
#     # To create features we read the image and extract the features using feature extractor
#     # To create labels we read them from the csv file
#     X_train, labels = [], []
#     for idx, (subject_id, label) in enumerate(csv_reader):
#         # Each loop is a line of the csv
#         print("Progress {:2.1%}".format((idx + 1) / n_samples), end="\r")
#         # Read image
#         img = io.imread(os.path.join('./data/train_sample/', subject_id + '.tif'))
#         img=rgb2gray(img)
#         # print(img.shape)
#         # Extract features
#         # X_train.append(my_feature_extractor(img))  # You can make some code to extract features.
#         # Read label
#         X_train.append(img)
#         labels.append(label)
#         if idx == (n_samples - 1) or idx == 8999:
#             break
#
#     # Creating features for test set
#     # labels are not available
#     csv_reader = csv.reader(open('./test_labels_sample.csv', 'r'))
#     X_test = []
#     for idx, subject_id in enumerate(csv_reader):
#         img = io.imread(os.path.join('./data/train_sample/', subject_id[0] + '.tif'))
#         img = rgb2gray(img)
#         # X_test.append(my_feature_extractor(img))
#         X_test.append(img)
#         if idx == min(n_samples - 1, 1000):
#             break
#
#     return np.array(X_train), np.array(labels), np.array(X_test)
# total_images = 8000 # dont go over 8000
#
# X_train, y_train, X_test = load_data_and_extract_features(total_images)
#
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
#
# # from tempfile import TemporaryFile
# # training_feature = TemporaryFile()
# np.save('training_feature.npy', X_train)
# np.save('training_label.npy', y_train)
# np.save('testing_feature.npy',X_test)

from sklearn.ensemble import RandomForestClassifier

import numpy as np
X_train=np.load('training_feature.npy')
y_train=np.load('training_label.npy')
X_test=np.load('testing_feature.npy')



from keras import optimizers
from sklearn.model_selection import train_test_split

import keras
num_classes=2
y_for_train = keras.utils.to_categorical(y_train, num_classes)


train, test, train_label, test_label = train_test_split(X_train, y_for_train, test_size=0.2 , random_state=42)

print(train.shape[0], 'train samples')
print(test.shape[0], 'validation samples')

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
epochs = 100
batch_size =10


# In[27]:


train = train.reshape(train.shape[0], 96, 96 , 1).astype('float32')
test = test.reshape(test.shape[0], 96, 96 , 1).astype('float32')


## to prevent overfitting: 1. increase the dataset size & variability 2. broaden the network 3. Reduce the depth 4. dropout

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
#create model
model = Sequential()
#add model layers


model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(96,96,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(16, kernel_size=3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(8, kernel_size=3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(4, kernel_size=3, activation="relu"))

model.add(Flatten())
model.add(Dense(num_classes, activation="softmax"))
#compile model using accuracy to measure model performance
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.0001,  epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])
# In[28]:


#train the model
model.fit(train, train_label, validation_data=(test, test_label), epochs=500)




#####################################################
# testing the model
final_test = X_test.reshape(-1, 96, 96 , 1).astype('float32')
final_prediction=model.predict_classes(final_test)
with open("submission.csv", 'w') as f:
    for i in final_prediction:
        p = str(i) + '\n'
        f.write(p)
