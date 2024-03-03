# -*- coding: utf-8 -*-
"""

@author: profa
"""
## Image Processing Python
## https://note.nkmk.me/en/python-numpy-image-processing/

#########################################################
##
## The **dataset for this code**
##
## #CIFAR-10 dataset consists of 60000 32x32 
## color images in 10 classes, with 6000 images per class
## https://www.cs.toronto.edu/~kriz/cifar.html
## https://keras.io/api/datasets/cifar10/
## How to read it in...
## (train_images, train_labels), (test_images, test_labels) ##=datasets.cifar10.load_data()
####################################################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import tensorflow.keras
#from tensorflow.keras.datasets import mnist
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import numpy as np
import pandas as pd

plt.rcParams["figure.figsize"] = (5,5)
#CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class
# https://www.cs.toronto.edu/~kriz/cifar.html
# https://keras.io/api/datasets/cifar10/
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

print(type(train_images))
print(train_images.shape)   ## 50000 rows, 32 by 32, depth 3 (because RGB) 
# (50000, 32, 32, 3)
plt.imshow(train_images[3])


## Set the input shape
input_shape=train_images.shape
print("The input shape for the training images is\n", input_shape) ## (50000, 32, 32, 3)
print("The input shape per image is\n", input_shape[1:]) ## (32, 32, 3)  ## The "3" is because this has 3 channel (RGB)
#color_channels refers to (R,G,B)
print("A single image, R of RGB, has a matrix like this:\n", train_images[0,:,:,0])
print("A single image has shape\n", train_images[0,:,:,0].shape)
print(train_images[0,:,:,2].shape)

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

## Print out  a visual of all the image categories................
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

##......................................................
#########################################################
## Build the CNN Model 
#########################################################

CNN_Model = tf.keras.models.Sequential([
    #https://keras.io/api/layers/convolution_layers/convolution2d/
  tf.keras.layers.Conv2D(input_shape=input_shape[1:], kernel_size=(3,3), filters=32, activation="relu"), 
  ## A CNN takes tensors of shape (image_height, image_width, color_channels)
        ## input_shape[1:] means all but the first value. Here, our input is: ()
        ## https://www.tensorflow.org/api_docs/python/tf/keras/activations
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  #https://keras.io/api/layers/pooling_layers/max_pooling2d/
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  #tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
  
  tf.keras.layers.Flatten(),
  
  tf.keras.layers.Dense(64, activation='relu'), 
  ## https://keras.io/api/layers/core_layers/dense/
  ## https://www.tutorialspoint.com/keras/keras_dense_layer.htm
  
  tf.keras.layers.Dense(10) 
])


CNN_Model.summary()


CNN_Model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              ## Using True above means you do not use one-hot-encoding
              metrics=['accuracy'])

##Increase epochs to improve accuracy/training
history = CNN_Model.fit(train_images, train_labels, epochs=15, 
                    validation_data=(test_images, test_labels))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = CNN_Model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

CNNpredictions=CNN_Model.predict([test_images])
print(CNNpredictions)
print(CNNpredictions.shape)

from sklearn.metrics import confusion_matrix
Pred_Max_Values = np.squeeze(np.array(CNNpredictions.argmax(axis=1)))
print(Pred_Max_Values)
CNN_CM=confusion_matrix(Pred_Max_Values, test_labels)
print(CNN_CM)

#########################################
## Pretty Confusion Matrix........................
#######################################
import seaborn as sns
import matplotlib.pyplot as plt     

fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(CNN_CM, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
#annot=True to annotate cells, ftm='g' to disable scientific notation
# annot_kws si size  of font in heatmap
# labels, title and ticks
ax.set_xlabel('Predicted labels') 
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix: NN') 
ax.xaxis.set_ticklabels(class_names,rotation=90, fontsize = 18)
ax.yaxis.set_ticklabels(class_names,rotation=0, fontsize = 18)