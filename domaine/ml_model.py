import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle   
import os
from csv import writer


def add_data(wine):
    """Add a wine to the data base

    Args:
        wine (WineModel): the wine we want to add
    """
    with open('datasource/Wines.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(wine)
        f_object.close()

def retrain_model():
    """Retrain the model with the current data
    """
    wines= pd.read_csv("datasource/Wines.csv")
    columns = wines.columns.tolist()
    columns = [c for c in columns if c not in ["quality","id"]]
    target = "quality"
    train = wines.sample(frac=0.8, random_state=1)
    test = wines.loc[~wines.index.isin(train.index)]
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
    model.fit(train[columns], train[target])
    predictions = model.predict(test[columns])
    filename = 'domaine/finalized_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    
def get_model_info():
    """Get current model info
    """
    info=[]
    #to do 
    return info

path="domaine/finalized_model.pkl"
#Train the model only if it's the first lunch
if not os.path.exists(path) :
    wines= pd.read_csv("datasource/Wines.csv")
    wines=X = wines.drop(columns=['Id'],axis=1)
    columns = wines.columns.tolist()
    columns = [c for c in columns if c not in ["quality"]]
    target = "quality"
    train = wines.sample(frac=0.8, random_state=1)
    test = wines.loc[~wines.index.isin(train.index)]
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
    model.fit(train[columns], train[target])
    predictions = model.predict(test[columns])
    filename = 'domaine/finalized_model.pkl'
    pickle.dump(model, open(filename, 'wb'))



