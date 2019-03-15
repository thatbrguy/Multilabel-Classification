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

## Custom callback to compute F1 Score and save the model
## with best validation F1 Score.

def F1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='samples')

class ComputeF1(Callback):
    
    def __init__(self):
        self.best_f1 = -1
        
    def on_epoch_end(self, epoch, logs={}):
        val_pred = np.round(self.model.predict(self.validation_data[0]))
        val_f1 = f1_score(self.validation_data[1], val_pred, average='samples')
        print('Validation Average F1 Score: ', val_f1)
        
        if val_f1 > self.best_f1:
            print('Better F1 Score, Saving model...')
            self.model.save('model.h5')
            self.best_f1 = val_f1


def load_data(df):
    
    trainX, testX, valX = [], [], []
    trainY, testY, valY = [], [], []
    
    for i in range(len(df)):
        
        item = df.loc[i][0]
        current_label = np.array((df.loc[i])[1:])
        
        path = os.path.join('images', item)
        list_of_imgs = [os.path.join(path, file) for file in os.listdir(path)]
        train_set = list_of_imgs[:30]
        val_set = list_of_imgs[30:40]
        test_set = list_of_imgs[40:]
        
        for file in train_set:
            img = cv2.resize(cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2RGB), (224, 224))
            trainX.append(img)
            trainY.append(current_label)
        
        for file in val_set:
            img = cv2.resize(cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2RGB), (224, 224))
            valX.append(img)
            valY.append(current_label)
        
        for file in test_set:
            img = cv2.resize(cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2RGB), (224, 224))
            testX.append(img)
            testY.append(current_label)
            
    return (np.array(trainX), np.array(trainY), np.array(testX), 
            np.array(testY), np.array(valX), np.array(valY))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices = ['ResNet50', 'Xception', 'DenseNet121', 'MobileNet'], 
                        help = 'Select a model to train', required = True)

    args = parser.parse_args()

    print('Loading Data...')
    df = pd.read_csv('clean_anno_reduced.csv')
    trainX, trainY, testX, testY, valX, valY = load_data(df)
    print('Data Loaded.')

    ## Normalization

    trainX = trainX.astype(np.float32)
    testX = testX.astype(np.float32)
    valX = valX.astype(np.float32)

    trainY = trainY.astype(np.float32)
    testY = testY.astype(np.float32)
    valY = valY.astype(np.float32)

    MEAN = np.mean(trainX, axis = (0,1,2))
    STD = np.std(trainX, axis = (0,1,2))

    for i in range(3):
        trainX[:, :, :, i] = (trainX[:, :, :, i] - MEAN[i]) / STD[i]
        testX[:, :, :, i] = (testX[:, :, :, i] - MEAN[i]) / STD[i]
        valX[:, :, :, i] = (valX[:, :, :, i] - MEAN[i]) / STD[i]

    f1_score_callback = ComputeF1()
    model = build_model('train', model_name = args.model)

    ## Training model.
    model.fit(trainX, trainY, batch_size = 32, epochs = 25, validation_data = (valX, valY), 
              callbacks = [f1_score_callback])

    ## Compute test F1 Score
    model = load_model('model.h5')

    score = F1_score(testY, model.predict(testX).round())
    print('F1 Score =', score)
