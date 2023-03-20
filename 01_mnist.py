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
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()


#numpy to numba
x_train_cp = cp.asarray(x_train, dtype=cp.float32)
x_valid_cp = cp.asarray(x_valid, dtype=cp.float32)


x_train_cp = cp.reshape(x_train_cp,  (x_train_cp.shape[0], x_train_cp.shape[1]*x_train_cp.shape[2]))
x_test_cp = cp.reshape(x_valid_cp,  (x_valid_cp.shape[0], x_valid_cp.shape[1]*x_valid_cp.shape[2]))#number of images, then the size of the image


#x_train_cp = np.reshape(x_train,  (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
#x_test_cp = np.reshape(x_valid,  (x_valid.shape[0], x_valid.shape[1]*x_valid.shape[2]))#number of images, then the size of the image


#normalize image
x_train_cp = x_train_cp/255.0
x_test_cp = x_test_cp/255.0

#this kind of transformation modifies the data so that each value is a collection of all possible categories, with the actual category that this particular value is set as #true.


num_categories = 10

y_train = keras.utils.to_categorical(y_train, num_categories)
y_valid = keras.utils.to_categorical(y_valid, num_categories)
# callbacks


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

csv_logger = tf.keras.callbacks.CSVLogger('training.log')

# training parameters
EPOCHS = 5
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

x_train_np = cp.asnumpy(x_train_cp)
x_validation_np = cp.asnumpy(x_test_cp)
# history = model.fit(x_train_cp.asnumpy(), y_train, validation_data=(
#     xtest_norm, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE,
#     verbose=2, shuffle=True)

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


#0.9907666444778442 0.9793999791145325




