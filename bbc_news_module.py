# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 20:39:44 2022

@author: Si Kemal
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.layers import Bidirectional, SpatialDropout1D
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras import Input
import matplotlib.pyplot as plt
import numpy as np

class ModelCreation():
    def __init__(self):
      pass
    
    def model_layer(self,num_node=128,drop_rate=0.2,output_node=5,embed_dims = 64,vocab_size = 10000):
        model=Sequential()
        model.add(Input(shape=(340))) #input_length #features 
        model.add(Embedding(vocab_size,output_dim=embed_dims))
        model.add(SpatialDropout1D(0.4))
        model.add(Bidirectional(LSTM(embed_dims,return_sequences=True))) # only once return_se=true when LSTM meet LSTM after
        model.add(Dropout(drop_rate))
        model.add(LSTM(num_node))
        model.add(Dropout(drop_rate))
        model.add(Dense(num_node, activation='relu'))
        model.add(Dropout(drop_rate))
        model.add(Dense(output_node, activation='softmax'))
        model.summary()
        return model

class Model_Evaluate():
    def __init__(self):
        pass
    
    def EvaluateMymodel(self,model,X_test,y_test):
        results = model.evaluate(X_test,y_test)
        pred_y = np.argmax(model.predict(X_test), axis=1)
        true_y = np.argmax(y_test,axis=1)
        
        cm = confusion_matrix(true_y,pred_y)
        cr = classification_report(true_y,pred_y)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        
        print(results)
        print(cm)
        print(cr)
