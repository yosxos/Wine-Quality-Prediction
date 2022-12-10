import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle   
import os
from csv import writer


def add_data(wine):
    """Add a wine to the data base

    Args:
        wine (WineModel): the wine we want to add
    """
    with open('/home/yasait/Wine-Quality-Prediction/datasource/Wines.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(wine)
        f_object.close()

def retrain_model():
    """Retrain the model with the current data
    """
    df= pd.read_csv("/home/yasait/Wine-Quality-Prediction/datasource/Wines.csv")
    X = df.drop(columns=['quality'],axis=1)
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    sc_x = StandardScaler()
    X_train = sc_x.fit_transform(X_train)
    X_test = sc_x.fit_transform(X_test)
    dTree_clf = DecisionTreeClassifier()
    dTree_clf.fit(X_train,y_train)
    filename = 'finalized_model.sav'
    pickle.dump(dTree_clf, open(filename, 'wb'))
def get_model_info():
    """Get current model info
    """
    info=[]
    #to do 
    return info

path="'domaine/finalized_model.pkl'"
#Train the model only if it's the first lunch
if os.path.exists(path)== False :
    df= pd.read_csv("/home/yasait/Wine-Quality-Prediction/datasource/Wines.csv")
    X = df.drop(columns=['quality'],axis=1)
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    sc_x = StandardScaler()
    X_train = sc_x.fit_transform(X_train)
    X_test = sc_x.fit_transform(X_test)
    dTree_clf = DecisionTreeClassifier()
    dTree_clf.fit(X_train,y_train)
    filename = 'domaine/finalized_model.pkl'
    pickle.dump(dTree_clf, open(filename, 'wb'))
    y_pred2 = dTree_clf.predict(X_test)
    print("Accuracy of Model1::",accuracy_score(y_test,y_pred2))
