# Import libraries for data wrangling, preprocessing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
from keras import backend as K
from keras import callbacks
from keras import layers
from keras import models
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.externals import joblib
# Importing libraries for building the neural network
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, ClassifierMixin



class FakeNoiseNN(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self, intValue=0, stringParam="defaultValue", otherParam=None):
        """
        Called when initializing the classifier
        """
        self.intValue = intValue
        self.stringParam = stringParam

        # THIS IS WRONG! Parameters should have same name as attributes
        self.differentParam = otherParam 
               
        self.directory = "../models/supervisedBC/"

    def train_KerasBinaryClassifier(self,X_train,y_train):

        # Use Tenserflow backend
        sess = tf.Session()
        K.set_session(sess)

        def model():
            model = models.Sequential([
                layers.Dense(128, input_dim=X_train.shape[1], activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
            ])
            model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            return model


        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

        pipe = pipeline.Pipeline([
            ('rescale', preprocessing.StandardScaler()),
            ('nn', KerasClassifier(build_fn=model, nb_epoch=10, batch_size=128,
                                   validation_split=0.2, callbacks=[early_stopping]))
        ])


        pipe.fit(X_train, y_train)

 
        model_step = pipe.steps.pop(-1)[1]
        joblib.dump(pipe, os.path.join(self.directory, 'pipeline.pkl'))
        print("Trained Model is Saved at relative path inside PROJECT_DIR ",self.directory)
        models.save_model(model_step.model, os.path.join(self.directory, 'model.h5'))
        return

    def fit(self,X_Pos,X_PosLabel,X_Neg,X_NegLabel):
        
        data = np.concatenate((X_Pos,X_Neg),axis=0)
        label = np.concatenate((X_PosLabel,X_NegLabel),axis=0)
        print("Training the Keras Binary classifier.....")
        self.train_KerasBinaryClassifier(data,label)
 
    def predict(self,X_testPos,X_testNeg):
        

        X_test = np.concatenate((X_testPos,X_testNeg),axis=0)
        X_testPosLabel = np.ones(len(X_testPos))
        X_testNegLabel = np.zeros(len(X_testNeg))
        y_test = np.concatenate((X_testPosLabel,X_testNegLabel),axis=0)
        pipe = joblib.load(os.path.join(self.directory, 'pipeline.pkl'))
        model = models.load_model(os.path.join(self.directory, 'model.h5'))
        pipe.steps.append(('nn', model))


        y_pred_keras = pipe.predict_proba(X_test)[:, 0]
        from sklearn.metrics import roc_curve
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
        from sklearn.metrics import auc
        auc_keras = auc(fpr_keras, tpr_keras)
        print(auc_keras)
        return auc_keras
 
    def score(self, X, y=None):
        # counts number of values bigger than mean
        print(" Score function is not implemented for FakeNN")
        return
      
      
        
