import os
import cv2
import argparse
import numpy as np
import pandas as pd

from collections import Counter

from keras.callbacks import Callback
from keras.backend import clear_session
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten
from keras.applications import ResNet50, MobileNet, Xception, DenseNet121

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from keras_model import build_model

if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    MEAN = np.array([51.072815, 51.072815, 51.072815])
    STD = np.array([108.75629,  92.98068,  85.61884])

    categories = [
    'healthy', 'junk', 'dessert', 'appetizer', 'mains', 'soups', 'carbs', 'protein', 'fats', 'meat'
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help = 'Path to the image to be predicted', required = True)
    parser.add_argument('--saved_model', help = 'Path of the saved model', required = True)

    args = parser.parse_args()

    model = build_model('inference', model_path = args.saved_model)

    img = np.expand_dims(cv2.imread(args.image, 1), axis = 0)

    for i in range(3):
        img[:, :, :, i] = (img[:, :, :, i] - MEAN[i]) / STD[i]

    prediction = np.round(model.predict(img)[0])
    labels = [categories[idx] for idx, current_prediction in enumerate(prediction) if current_prediction == 1]

    print('Prediction:', labels)

