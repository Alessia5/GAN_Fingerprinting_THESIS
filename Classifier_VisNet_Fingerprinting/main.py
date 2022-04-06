# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import *
from sklearn.metrics import confusion_matrix, classification_report

import glob
import json

import os

import tensorflow as tf
from keras.layers import LeakyReLU, Embedding, Masking, LSTM
from tensorflow import keras

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Reshape, Conv2DTranspose, UpSampling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras

from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import optimizers
from keras.models import load_model

import matplotlib.pyplot as plt



import config
import generator_NO_AUG
import PreDownsampling_residual
import PreDownsampling
import PostPooling
import Autoencoder
print('ciao')

print('Data set Training elements:')
print('ATTGAN \t' + str(len(os.listdir(config.DS_Path + 'ATTGAN'))))
print('CELEBA \t' + str(len(os.listdir(config.DS_Path + 'CELEBA'))))
print('GDWCT \t' + str(len(os.listdir(config.DS_Path + 'GDWCT'))))
print('STARGAN \t' + str(len(os.listdir(config.DS_Path + 'STARGAN'))))
print('STYLEGAN \t' + str(len(os.listdir(config.DS_Path + 'STYLEGAN'))))
print('STYLEGAN2 \t' + str(len(os.listdir(config.DS_Path + 'STYLEGAN2'))))

print('Data set Testing elements:')
print('ATTGAN \t' + str(len(os.listdir(config.TE_Path + 'ATTGAN'))))
print('CELEBA \t' + str(len(os.listdir(config.TE_Path + 'CELEBA'))))
print('GDWCT \t' + str(len(os.listdir(config.TE_Path + 'GDWCT'))))
print('STARGAN \t' + str(len(os.listdir(config.TE_Path + 'STARGAN'))))
print('STYLEGAN \t' + str(len(os.listdir(config.TE_Path + 'STYLEGAN'))))
print('STYLEGAN2 \t' + str(len(os.listdir(config.TE_Path + 'STYLEGAN2'))))


def savemodel(model, problem):
    filename = os.path.join(config.Models_Dir_Path, config.Model_Name + '.h5')
    model.save(filename)
    print("\nModel saved successfully on file %s\n" % filename)


def loadmodel(problem):
    filename = os.path.join(config.Models_Dir_Path, '%s.h5' % problem)
    try:
        model = load_model(filename)
        print("\nModel loaded successfully from file %s\n" % filename)
    except OSError:
        print("\nModel file %s not found!!!\n" % filename)
        model = None
    return model

    ## Train and Test the model ##





## Plotting the results ##
def show_train_performance(history):
    # Accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    # Loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()





### TRAINING AND VALIDATION PHASE ###
steps_per_epoch = generator_NO_AUG.TR_Gen.n//generator_NO_AUG.TR_Gen.batch_size
val_steps = generator_NO_AUG.Val_Gen.n//generator_NO_AUG.Val_Gen.batch_size+1
epochs = 10

checkpoint_filepath = config.Model_Path2
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
try:
    history = Autoencoder.model.fit(generator_NO_AUG.TR_Gen, epochs=epochs, callbacks=[model_checkpoint_callback], verbose=1,\
                    steps_per_epoch=steps_per_epoch,\
                    validation_data=generator_NO_AUG.Val_Gen,\
                    validation_steps=val_steps)
    #show_train_performance(history)
except KeyboardInterrupt:
    pass


# Loading and showing the model
show_train_performance(history)
# The model weights (that are considered the best) are loaded into the model
Autoencoder.model.load_weights(checkpoint_filepath)
print("\nModel loaded successfully on file %s\n" %checkpoint_filepath)


### TESTING PHASE ###
generator_NO_AUG.TE_Gen = generator_NO_AUG.test_datagen.flow_from_directory(
    directory=config.TE_Path,  # TE_Path
    target_size=(128, 128),
    color_mode="rgb",
    batch_size=generator_NO_AUG.batch_size,
    class_mode="categorical",
    shuffle=False

)
val_steps = generator_NO_AUG.TE_Gen.n // generator_NO_AUG.TE_Gen.batch_size + 1
loss, acc = Autoencoder.model.evaluate(generator_NO_AUG.TE_Gen, verbose=1, steps=val_steps)
print('Test loss: %f' % loss)
print('Test accuracy: %f' % acc)


### Showing confusion matrix and classification final report ###
import sklearn.metrics
from sklearn.metrics import classification_report, confusion_matrix

val_steps = generator_NO_AUG.TE_Gen.n//generator_NO_AUG.TE_Gen.batch_size+1

generator_NO_AUG.TE_Gen = generator_NO_AUG.test_datagen.flow_from_directory(
    directory=config.TE_Path,
    target_size=(128, 128),
    color_mode="rgb",
    batch_size=generator_NO_AUG.batch_size,
    class_mode="categorical",
    shuffle=False
)

preds = Autoencoder.model.predict(generator_NO_AUG.TE_Gen,verbose=1,steps=val_steps)

Ypred = np.argmax(preds, axis=1)
Ytest = generator_NO_AUG.TE_Gen.classes  # shuffle=False in TE_Gen

print(confusion_matrix(Ytest, Ypred))
print(classification_report(Ytest, Ypred, labels=None, target_names=generator_NO_AUG.classnames, digits=3))