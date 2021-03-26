from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.staticfiles.storage import staticfiles_storage

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def home(request):
    return render(request, 'index.html', {"predicted": ""})

def predict(request):

    sl = float(request.GET['sl'])
    sw = float(request.GET['sw'])
    pl = float(request.GET['pl'])
    pw = float(request.GET['pw'])

    rawdata = staticfiles_storage.path('Iris.csv')
    dataset = pd.read_csv(rawdata)

    x = dataset.iloc[:,1:5].values
    y = dataset.iloc[:,5].values

    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8 , random_state = 27)

    model = DecisionTreeClassifier()
    model.fit(x_train,y_train)

    ip = np.array([[sl, sw, pl, pw]])

    y_pred = model.predict(ip)
    
    #model.predict([[5.7,2.6,3.5,1.0]])

    return render(request, 'index.html', {'predicted': y_pred, 'sl': sl, 'sw': sw, 'pl': pl, 'pw': pw})