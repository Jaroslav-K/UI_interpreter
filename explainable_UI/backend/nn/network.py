

from tensorflow import keras
from matplotlib import pyplot as plt
from keras import backend
import numpy as np
import tensorflow.keras.datasets.mnist as input_data
import random
import nn.decoder as decoder
import pandas as pd
from PIL import Image


model = keras.models.load_model('LeNet5Model_28input.h5')

layer = model.get_layer('fully_connected_3')
last_layer = model.get_layer('output')

model_fully_connected = keras.Model(inputs=model.input, outputs=layer.output)
model_last_layer = keras.Model(inputs=last_layer.input, outputs=last_layer.output)

features = []
# default prediction
prediction = np.expand_dims(np.zeros(10), axis=0)

model_min = -4.965
model_max = 6.332

model_mean = 33.351
model_std = 78.619


def upload_image(img_path):
    global features
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)[:, :, :1]
    img = np.expand_dims(img, axis=0)
    img = prepare_data(img)
    # features
    features = extract_features(img)
    # layer prediction
    layer_prediction(features)
    # encoder
    decoder.predict(features)
    return True


# img = (img - mean)/std
def prepare_data(img_array):
    """
    Prepare input image data by standardization. (img - model_mean)/std_mean
    :param img_array: input image array
    :return: standardized image
    """
    return (img_array - model_mean)/model_std


def img_to_array():
    """
    Transform image to numpy array.
    :return: image array
    """
    img = keras.preprocessing.image.load_img('data.jpg')
    img = keras.preprocessing.image.img_to_array(img)[:, :, :1]
    img = np.expand_dims(img, axis=0)
    return img


def get_prediction():
    """
    Get prediction.
    :return: prediction
    """
    global prediction
    return prediction


def get_features():
    """
    Get image features values.
    :return: features values from last layer
    """
    return features[0]


def update_features(data):
    """
    Update features values.
    :param data: changed features
    :return: True/False
    """
    global features

    features[0] = data
    # update prediction
    layer_prediction(features)
    # update encoder
    decoder.predict(features)
    return True


def normalize(array):
    """
    Function to normalize input array by model (MIN-MAX).
    :param array: input array
    :return: normalized array
    """
    return (array - model_min)/(model_max - model_min)


def denormalize(array):
    """
    Function to denormalize input array to original values by model (MIN-MAX).
    :param array: input array
    :return: denormalized array
    """
    return array * abs(model_max - model_min) + model_min


def get_model_prediction(img_array):
    """
    Get model prediction. Return classification of input image.
    :param img_array: input image
    :return: classification of image
    """
    return model.predict(img_array)


def extract_features(img_array):
    """
    Extract features from last fully connected layer of size 84.
    :param img_array: input image
    :return: array of 84 features
    """
    return model_fully_connected.predict(img_array)


def layer_prediction(img_features):
    """
    Return prediction of last layer. Prediction is classification of model.
    :param img_features: image features of size 84
    :return: prediction of size 10
    """
    global prediction
    prediction = model_last_layer.predict(img_features)
    return prediction

