import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist
import cudf 
import cuml
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
"""With the `mnist` module, we can easily load the MNIST data, already partitioned into images and labels for both training and validation:"""

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






start_time = time.time()
# the data, split between train and validation sets
training = cudf.read_csv("/media/gkasap/ssd256gb/DLintro/sign_mnist_train.csv", dtype = cp.float32) 
valid = cudf.read_csv("/media/gkasap/ssd256gb/DLintro/sign_mnist_valid.csv", dtype = cp.float32)

x_training, y_training = training.iloc[:,1:], training.iloc[:,0]
x_valid, y_valid = valid.iloc[:,1:], valid.iloc[:,0]
#sanity check x_traing.max().max(), x_traing.min().min(), x_valid.max().max(), x_valid.min().min()

#normalize image
x_training = x_training/255.0
x_valid = x_valid/255.0

y_training_pd = y_training.to_pandas()
y_valid_pd = y_valid.to_pandas()
#this kind of transformation modifies the data so that each value is a collection of all possible categories, with the actual category that this particular value is set as #true.
num_categories = 24
y_train = keras.utils.to_categorical(y_training_pd, num_categories)
y_valid = keras.utils.to_categorical(y_valid_pd, num_categories)
# callbacks


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

csv_logger = tf.keras.callbacks.CSVLogger('training.log')

# training parameters
EPOCHS = 25
BATCH_SIZE = 64
METRICS = [ "accuracy"]
OPTIMIZER = keras.optimizers.Adam(learning_rate=0.001)


# Create the model
CALLBACKS = [early_stopping, csv_logger]
model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units = 512, activation='relu'))
model.add(Dense(units=num_categories, activation='softmax', 
                          bias_initializer=tf.keras.initializers.Constant(-np.log(1.0/num_categories))))
          
model.compile(loss='categorical_crossentropy', metrics=METRICS, optimizer=OPTIMIZER)

#cudf to pandas
x_train_np = x_training.to_pandas()
x_validation_np = x_valid.to_pandas()

history = model.fit(
    x_train_np, y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    verbose=1, 
    shuffle=True,
    validation_data=(x_validation_np, y_valid)
    callbacks=CALLBACKS
)




plot_history(history)
#print history accuracy
print(history.history['accuracy'][-1], history.history['val_accuracy'][-1])






