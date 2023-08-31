from tensorflow import keras
import numpy as np

model = keras.models.load_model('decoder_v2.h5')


def predict(features):
    """
    Function to create reconstructed image from features extracted from classification model last layer.
    Image is saved on disk.
    :param features: features from classification model.
    :return:
    """
    data = model.predict(features)[0]
    img = keras.utils.array_to_img(data, scale=True)
    img.save('reconstructed.png')


