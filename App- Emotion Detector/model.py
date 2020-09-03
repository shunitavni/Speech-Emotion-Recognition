import librosa  # Python Library for analysing audio
import self as self
import soundfile
import os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as functional
# import torch.nn as nn
# import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets._base import load_data
from sklearn.model_selection import train_test_split
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
import pandas as pd

class Model:
    def __init__(self):
        self.init = True

    def extract_features(self, file_name, mfcc, chroma, mel, contrast, tonnetz):
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate

            if chroma:
                stft = np.abs(librosa.stft(X))
                result = np.array([])
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))
            if chroma:
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma))
            if mel:
                mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mel))
            if contrast:
                contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, contrast))
            if tonnetz:
                tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
                result = np.hstack((result, tonnetz))

        return result

    def load_data(self, test_size, happy, anger, neural, sad):
        X, y = [], []
        #   # if csv :#:
        #     for file in spects:
        #         file_name = file[0]
        #         emotion = file[1]
        #         feature=extract_features(file_name, mfcc = True, chroma = True, mel = True, contrast= True ,tonnetz= True)
        #         X.append(feature)
        #         y.append(emotion)

        #:

        if happy:
            with open('csv/f_happy.csv') as csvDataFile:
                csvReader = csv.reader(csvDataFile, quoting=csv.QUOTE_NONNUMERIC)
                for row in csvReader:
                    X.append(row)

            with open('csv/labels_happy.csv') as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                for row in csvReader:
                    y.append(row)

        if anger:
            with open('csv/f_anger.csv') as csvDataFile:
                csvReader = csv.reader(csvDataFile, quoting=csv.QUOTE_NONNUMERIC)
                for row in csvReader:
                    X.append(row)

            with open('csv/labels_anger.csv') as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                for row in csvReader:
                    y.append(row)

        if sad:
            with open('csv/f_sad.csv') as csvDataFile:
                csvReader = csv.reader(csvDataFile, quoting=csv.QUOTE_NONNUMERIC)
                for row in csvReader:
                    X.append(row)

            with open('csv/labels_sad.csv') as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                for row in csvReader:
                    y.append(row)

        if neural:
            with open('csv/f_neutral.csv') as csvDataFile:
                csvReader = csv.reader(csvDataFile, quoting=csv.QUOTE_NONNUMERIC)
                for row in csvReader:
                    X.append(row)

            with open('csv/labels_neutral.csv') as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                for row in csvReader:
                    y.append(row)

        return train_test_split(np.array(X), y, test_size=test_size, random_state=9)

#########################################

    def train_classifiers(self):

        ##Classifiers are trained to be used below :
        # 1. clasiffier between happy and anger

        X_train_1, X_test_1, y_train_1, y_test_1 = self.load_data(test_size=0.25, happy=True, anger=True, neural=False,
                                                             sad=False)
        self.classifier1 = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,),
                                    learning_rate='adaptive', max_iter=500)
        self.classifier1.fit(X_train_1, y_train_1)
        #y_pred_1 = classifier1.predict(X_test_1)

        # 2. clasiffier between neutral & sad
        X_train_2, X_test_2, y_train_2, y_test_2 = self.load_data(test_size=0.25, happy=False, anger=False, neural=True,
                                                             sad=True)

        self.classifier2 = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,),
                                    learning_rate='adaptive', max_iter=500)
        self.classifier2.fit(X_train_2, y_train_2)
        #y_pred_2 = classifier2.predict(X_test_2)

        # classifier between happy+angry--0  VS.  Neu+sad--1
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(test_size=0.25, happy=True, anger=True, neural=True, sad=True)

        self.y_train_bin, self.y_test_bin = [], []  # they binarys & y_train, y_test regulars

        [self.y_train_bin.append(0) if self.y_train[i][0] == 'happy' or self.y_train[i][0] == 'anger' else self.y_train_bin.append(1) for i
         in range(len(self.y_train))]
        [self.y_test_bin.append(0) if self.y_test[i][0] == 'happy' or self.y_test[i][0] == 'anger' else self.y_test_bin.append(1) for i in
         range(len(self.y_test))]

        self.classifier = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,),
                                   learning_rate='adaptive', max_iter=500)

        self.classifier.fit(self.X_train, self.y_train_bin)
        self.y_pred = self.classifier.predict(self.X_test)

    def train_hierarchical(self):
        X_hir = []
        y_true, y_hir_pred, y_hir_bin = [], [], []

        # collect all the example to be test
        [X_hir.append(self.X_train[i]) for i in range(len(self.X_train))]
        [X_hir.append(self.X_test[i]) for i in range(len(self.X_test))]

        [y_true.append(self.y_train[i]) for i in range(len(self.y_train))]
        [y_true.append(self.y_test[i]) for i in range(len(self.y_test))]

        [y_hir_bin.append(self.y_train_bin[i]) for i in range(len(self.y_train_bin))]
        [y_hir_bin.append(self.y_test_bin[i]) for i in range(len(self.y_test_bin))]

        y_hir_pred = self.classifier.predict(X_hir)  # 1 classifier

        # layer 2 --- all happy & anger goto classifier 1 ,
        X1 = []  # 0 - happy, anger - only examples
        X2 = []  # 1- ...
        y_true_1 = []
        y_true_2 = []
        y_allPreds, y_allTrue = [], []

        for i in range(len(y_hir_pred)):
            if y_hir_pred[i] == 0:
                X1.append(X_hir[i])
                y_true_1.append(y_true[i])
            else:
                X2.append(X_hir[i])
                y_true_2.append(y_true[i])

        y_pred_finally = self.classifier1.predict(X1)
        result = self.classifier1.predict_proba(X1)
        # print(result)

        [y_allPreds.append(y_pred_finally[i]) for i in range(len(y_pred_finally))]
        [y_allTrue.append(y_true_1[i]) for i in range(len(y_true_1))]

        y_pred_finally = self.classifier2.predict(X2)
        result2 = self.classifier2.predict_proba(X2)
        # print(y_pred_finally)

        [y_allPreds.append(y_pred_finally[i]) for i in range(len(y_pred_finally))]
        [y_allTrue.append(y_true_2[i]) for i in range(len(y_true_2))]

        accuracy = accuracy_score(y_true=y_allTrue, y_pred=y_allPreds)
        print("Accuracy: {:.2f}%".format(accuracy * 100))
    # classes = ('anger', 'happy', 'neutral', 'sad')

    # target_names = classes
    # print(classification_report(y_test,y_pred, target_names=target_names))

    def predict_new_audio(self, *new_ex):
      stat=[]
      y_pred=self.classifier.predict(new_ex)
      result=self.classifier.predict_proba(new_ex)
      # print(result)

      if(y_pred==0):
        y_pred_finally=self.classifier1.predict(new_ex)
        result1=self.classifier1.predict_proba(new_ex)
        stat.append(round(result1[0][0] * 100, 2))
        stat.append(round(result1[0][1] * 100, 2))
        others = (result[0][1]/2)
        stat.append(round(others * 100, 2))
        stat.append(round(others * 100, 2))
        # print(result)

      else:
        y_pred_finally=self.classifier2.predict(new_ex)
        result2=self.classifier2.predict_proba(new_ex)
        others = (result[0][0] / 2)
        stat.append(round(others * 100, 2))
        stat.append(round(others * 100, 2))

        stat.append(round(result2[0][0] * 100, 2))
        stat.append(round(result2[0][1] * 100, 2))

        # print(result)
      return y_pred_finally, stat

    def newAudio(self, file):
        feature = self.extract_features(file, mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True)
        label_new, stat = self.predict_new_audio(feature)
        stat20 = []
        [stat20.append(stat[i] + 20) for i in range(len(stat))]
        print(stat)
        print(stat20)
        return label_new[0], stat, stat20



################################################################

# model = Model()
# model.train_classifiers()
# model.train_hierarchical()
