#!/usr/bin/env python
# coding: utf-8

# # 1 - Explication du sujet

# Ce projet consiste à utiliser le pouvoir explicatif d'un nombre important de variables du mix éléctrique, des interconnexions, et météorologiques de la France et de l'Allemagne afin de pouvoir expliquer la variation des prix forward 24h de l'éléctricité en France. Ce projet ne consiste pas à effectuer de prédiction, les prix et les données explicatives proviennent du même jour.
# Premièrement appliquer un algorithme d'apprentissage non supervisé sur les données puis nous allons utiliser un algorithme de RandomForest afin de préduire de manière la plus precise possible la variable cible. Enfin nous allons utiliser un réseau de neurone et comparer les résultats

import pandas as pd
import os 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np

warnings.filterwarnings('ignore')


# # 2 - Import des données et traitement des données

def pre_traitement_données(X, y):
    data = pd.concat([X, y], axis=1)
    data = data[data['COUNTRY']=='FR'].reset_index(drop=True)
    X = data.drop(['ID', 'DAY_ID', 'TARGET', 'COUNTRY'], axis=1)
    y = data['TARGET']
    return X, y


def mean_absolute_error(y_pred, y_true):
    """
    Calculates the mean absolute error between two arrays of predictions and target values.
    """
    mae = np.mean(np.abs(y_pred - y_true))
    return mae



# get the current working directory
current_dir = os.getcwd()

X_train = pd.read_csv(current_dir + "\\X_train_NHkHMNU.csv")
y_train = pd.read_csv(current_dir + "\\y_train_ZAN5mwg.csv")

X_test = pd.read_csv(current_dir + "\\X_test_final.csv")
y_test = pd.read_csv(current_dir + "\\y_test_random_final.csv")


X_train, y_train = pre_traitement_données(X_train, y_train)
X_test, y_test = pre_traitement_données(X_test, y_test)


X_train.head()

X_train

y_train.head()

X_train.describe()


# L'ensemble des variables ont été normalisé, elles ont donc une moyenne autour de 0 et un écart type autour de 1.
# De la même manière les quartiles sont proches

y_train.describe()

y_train.hist(bins=100)

#NA analysis on X_train 
sns.heatmap(X_train.isna())

# insert your code here
plt.figure(figsize=(16,4))
(len(X_train.index)-X_train.count()).plot.bar()


# Certaines variables possèdent des valeurs na que nous allons remplacer par la valeur moyenne de la colonne

## Fill the nans with mean value
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())