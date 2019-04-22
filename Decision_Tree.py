#!/usr/bin/env python
# coding: utf-8

import pandas as pd

df = pd.read_csv("Heart.csv")
#df.head()

df.isnull().sum()

df = df.dropna()

df.isnull().sum()

df = df.drop(["Unnamed: 0"],1)


#df.head()

#df.shape

df["AHD"].value_counts()

import numpy as np
numerical = df.select_dtypes(include = np.number)
nonnumerical = df.select_dtypes(exclude = np.number)
nonnumerical["Ca"] = numerical["Ca"]
numerical = numerical.drop(["Ca"],1)


from sklearn.preprocessing import MinMaxScaler


cols = list(numerical)


#cols

scaler = MinMaxScaler()

numerical = scaler.fit_transform(numerical)


numerical = pd.DataFrame(numerical)


numerical.columns = cols


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for x in list(nonnumerical.iloc[:,:-1]):
    nonnumerical[x] = le.fit_transform(nonnumerical[x])

nonnumerical.isnull().sum()
df = pd.concat([numerical,nonnumerical],1)

df = df.dropna()

#df.isnull().sum()

y = df["AHD"]
X = df.drop(["AHD"],1)

from sklearn.tree import DecisionTreeClassifier


dtc = DecisionTreeClassifier()
from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = tts(X,y, test_size = 0.2, random_state = 42)


dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)


