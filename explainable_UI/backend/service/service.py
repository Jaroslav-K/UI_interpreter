from nn import network, decoder
from objects import feature
import numpy as np
import functools


def get_prediction():
    """
    Function return prediction from neural network.
    :return: prediction values for each classification
    """
    result = [p.item() for p in network.get_prediction()[0]]
    return result


def update_features(data):
    """
    Function to update features of last layer.
    :param data: updated features from UI
    :return: result True/False
    """
    arr = _convert_to_array(data)
    arr = np.array(arr)
    denormalized = network.denormalize(arr)
    result = network.update_features(denormalized)
    return result


def get_features():
    """
    Function to get normalized features from last uploaded images.
    :return: list of normalized features
    """
    if len(network.features) == 0:
        return []
    else:
        features = network.get_features()
        normalized = network.normalize(features)
        features = _convert_to_feature(normalized)
        return sorted(features, key=functools.cmp_to_key(feature.compare_features_by_distance), reverse=True)


def upload_image(img_path):
    """
    Function upload image to neural network, to extract features and predict.
    :param img_path: path of image on disk
    :return: True if successfully uploaded, False otherwise
    """
    result = network.upload_image(img_path)
    return result


def _convert_to_feature(array):
    features = [feature.Feature(i, v.item()) for i, v in enumerate(array)]
    return features


def _convert_to_array(data):
    arr = [0] * len(data)
    for v in data:
        arr[v['id']] = float(v['value'])
    return arr
