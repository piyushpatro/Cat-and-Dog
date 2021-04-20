# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 19:53:28 2020

@author: priya
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import shutil
import random
import glob

tf.config.experimental.list_physical_devices('GPU')

os.chdir('C:/Users/priya/OneDrive/Desktop/Piyush/Neural networks/dogs-vs-cats')
if os.path.isdir('train/dogs') is False:
    os.makedirs('train/dogs')
    os.makedirs('train/cats')
    os.makedirs('test')
    os.makedirs('valid/cats')
    os.makedirs('valid/dogs')

os.chdir('C:/Users/priya/OneDrive/Desktop/Piyush/Neural networks/dogs-vs-cats')
for c in random.sample(glob.glob('dog*'), 1000):
    shutil.move(c, 'C:/Neural networks/dogs-vs-cats/train/dogs')
for c in random.sample(glob.glob('dog*'), 200):
    shutil.move(c, 'C:/Neural networks/dogs-vs-cats/valid/dogs')
for c in random.sample(glob.glob('dog*'), 50):
    shutil.move(c, 'C:/Neural networks/dogs-vs-cats/test')

os.chdir('C:/Users/priya/OneDrive/Desktop/Piyush/Neural networks/dogs-vs-cats')
for c in random.sample(glob.glob('cat*'), 1000):
    shutil.move(c, 'C:/Neural networks/dogs-vs-cats/train/cats')
for c in random.sample(glob.glob('cat*'), 200):
    shutil.move(c, 'C:/Neural networks/dogs-vs-cats/valid/cats')
for c in random.sample(glob.glob('cat*'), 50):
    shutil.move(c, 'C:/Neural networks/dogs-vs-cats/test')

os.chdir('C:/Users/priya/OneDrive/Desktop/Piyush/Neural networks/dogs-vs-cats')    
train_path='C:/Users/priya/OneDrive/Desktop/Piyush/Neural networks/dogs-vs-cats/train'
valid_path='C:/Users/priya/OneDrive/Desktop/Piyush/Neural networks/dogs-vs-cats/valid'
test_path='C:/Users/priya/OneDrive/Desktop/Piyush/Neural networks/dogs-vs-cats/test'

train_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['dogs', 'cats'], batch_size=10)
valid_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['dogs', 'cats'], batch_size=10)
test_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=test_path, target_size=(224, 224), batch_size=10, shuffle=True)


imgs, labels=next(train_batches)

def plot_images(images_arr):
    fig, axes = plt.subplot(1, 10, figsize=(20,20))
    axes=axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tightlayout()
    plt.show()

assert train_batches.n==2000
plot_images(imgs)

print(labels)

model=Sequential([Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(224, 224, 3)), 
                  MaxPool2D(pool_size=(2,2), strides=2), 
                  Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'), 
                  MaxPool2D(pool_size=(2,2), strides=2),
                  Flatten(), 
                  Dense(units=2, activation='softmax')])

print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)


VGG_16_model=tf.keras.applications.vgg16.VGG16()

print(VGG_16_model.summary())

model_new=Sequential()

for layers in VGG_16_model.layers[:-1]:
    model_new.add(layers)

print(model_new.summary())

for layers in model_new.layers:
    layers.trainable=False

model_new.add(Dense(units=2, activation='softmax'))
print(model_new.summary())

model_new.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
valid_imgs, valid_labels=next(valid_batches)

model_new.fit(x=train_batches, validation_data=valid_batches, epochs=3, verbose=2)
predictions=model_new.predict(test_batches)

cm=confusion_matrix(test_batches.classes, np.argmax(predictions, axis=1))

print(test_batches.class_indices)
print(cm)