### Importation needed libraries ###

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
#print("Libraries imported.")

def generate_GAN_Classifier_VisNet_Fingerprinting_PostPooling(input_shape):
    model = Sequential()
    # network d
    model.add(
        Conv2D(16, kernel_size=(7, 7), activation='relu', input_shape=(128, 128, 3), padding='same'))  # (128x128x16)
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))  # (64x64x32)
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))  # (64x64x512)
    model.add(Conv2D(6, kernel_size=(3, 3), activation='relu', padding='same'))  # (64x64x6)
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(6, activation='softmax'))  # #Labels

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def generate_GAN_Classifier_VisNet_Fingerprinting_PostPooling_NoPooling(input_shape):
    model = Sequential()
    # network d
    model.add(
        Conv2D(16, kernel_size=(7, 7), activation='relu', input_shape=(128, 128, 3), padding='same'))  # (128x128x16)

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))  # (64x64x32)

    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))  # (64x64x512)
    model.add(Conv2D(6, kernel_size=(3, 3), activation='relu', padding='same'))  # (64x64x6)
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(6, activation='softmax'))  # #Labels

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# create the model
input_shape = (128, 128, 3)
model = generate_GAN_Classifier_VisNet_Fingerprinting_PostPooling(input_shape)
model.summary()    