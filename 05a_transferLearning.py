import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import cudf 
import cuml
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image as image_utils
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

def readable_prediction(image_path):
    # Show image
    show_image(image_path)
    # Load and pre-process image
    image = load_and_process_image(image_path)
    # Make predictions
    predictions = model.predict(image)
    # Print predictions in readable form
    print('Predicted:', decode_predictions(predictions, top=3))


def load_and_process_image(image_path):
    # Print image's original shape, for reference
    print('Original image shape: ', mpimg.imread(image_path).shape)#1200, 1800, 3
    
    # Load in the image with a target size of 224, 224
    #Prefer loading data with tf.keras.utils.image_dataset_from_directory
    image = tf.keras.utils.load_img(image_path, target_size = (224,224))
    image = tf.keras.utils.img_to_array(image)
    # Convert the image from a PIL format to a numpy array
    #image = image_utils.img_to_array(image)
    # Add a dimension for number of images, in our case 1
    image = image.reshape(1,224,224,3)
    # Preprocess image to align with original ImageNet dataset
    image = preprocess_input(image)
    #ome models use images with values ranging from 0 to 1. Others from -1 to +1. Others use the "caffe" style, that is not normalized
    # Print image's shape after processing
    print('Processed image shape: ', image.shape)
    return image


def show_image(image_path):
    image = mpimg.imread(image_path)
    print(image.shape)
    plt.imshow(image)

# load the VGG16 network *pre-trained* on the ImageNet dataset
#https://keras.io/api/applications/#available-models
model = VGG16(weights="imagenet")


print(model.summary())

readable_prediction("/media/gkasap/ssd256gb/DLintro/happy_dog.jpg")
readable_prediction("/media/gkasap/ssd256gb/DLintro/sleepy_cat.jpg")
readable_prediction("/media/gkasap/ssd256gb/DLintro/brown_bear.jpg")

#np.argmax(preds) <= 268: