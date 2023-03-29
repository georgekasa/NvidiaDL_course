from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.applications.vgg16 import preprocess_input

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from keras.preprocessing.image import ImageDataGenerator
import cupy as cp
import time 





def plot_history(history):
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


base_model = keras.applications.VGG16(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top= False)

base_model.trainable = False

#https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16
#Note: each Keras Application expects a specific kind of input preprocessing. For VGG16, call tf.keras.applications.vgg16.
# preprocess_input on your inputs before passing them to the model. vgg16.preprocess_input will #convert the input images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset, #without scaling.






numberOfcategories = 6
inputs = keras.Input(shape=(224, 224, 3))
# Separately from setting trainable on the model, we set training to False 
x = base_model(inputs, training=False)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(512, activation = "relu", bias_initializer=keras.initializers.Constant(value=0.001))(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(numberOfcategories, activation = "softmax")(x)
model = keras.Model(inputs, outputs)


############################################################################################################
###################### set up call backs####################################################################
############################################################################################################
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

csv_logger = tf.keras.callbacks.CSVLogger('/home/gkasap/Desktop/training1.log')


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='/home/gkasap/Desktop/my_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
    )
# Saves the current weights after every epoch
CALLBACKS = [early_stopping, csv_logger,  model_checkpoint]
############################################################################################################
# training parameters
EPOCHS = 5
BATCH_SIZE = 32
METRICS = [ "accuracy"]
OPTIMIZER = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss = "categorical_crossentropy" , metrics = METRICS, optimizer = OPTIMIZER)

############################################################################################################
########ImageDataGenerator##################################################################################
############################################################################################################
datagen = ImageDataGenerator(
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False, # Don't randomly flip images vertically
    preprocessing_function = preprocess_input
)  

train_it = datagen.flow_from_directory('/media/gkasap/ssd256gb/datasets/fruitsDL/dataset/train', 
                                       target_size=(224, 224), 
                                       color_mode='rgb', 
                                       class_mode='categorical', 
                                       batch_size=BATCH_SIZE,
                                       seed=32)

# load and iterate validation dataset
valid_it = datagen.flow_from_directory('/media/gkasap/ssd256gb/datasets/fruitsDL/dataset/test', 
                                      target_size=(224, 224), 
                                      color_mode='rgb', 
                                      class_mode='categorical', 
                                      batch_size=BATCH_SIZE,
                                      seed=32)

history = model.fit(train_it,
          validation_data=valid_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=valid_it.samples/valid_it.batch_size,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          callbacks=CALLBACKS,
          verbose=1)
#x = keras.layers.Dense(512, activation='relu', kernel_initializer='glorot_uniform', bias_initializer=initializers.Constant(value=0.001))(x)


plot_history(history)


# for layer in base_model.layers[-4:]:
#     layer.trainable = True





