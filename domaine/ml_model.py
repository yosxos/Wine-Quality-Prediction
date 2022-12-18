import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle   
import os
import dotenv
from csv import writer

path="domaine/finalized_model.pkl"
dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)
#Train the model only if it's the first lunch
if not os.path.exists(path) :
    df=pd.read_csv("datasource/Wines.csv")
    df=df.drop(["Id"],axis=1)
    X = df.drop(['quality'], axis = 1)
    y = df['quality']
    # Normalize feature variables
    X_features = X
    X = StandardScaler().fit_transform(X)
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=0)
    model = RandomForestClassifier(random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rnd_score = model.score(X_test,y_test)
    rnd_MSE = mean_squared_error(y_test,y_pred)
    os.environ["Error"]=str(rnd_MSE)
    os.environ["Accuracy"]=str(rnd_score)
    dotenv.set_key("domaine/.env", "Error", os.environ["Error"])
    dotenv.set_key("domaine/.env", "Accuracy", os.environ["Accuracy"])
    filename = 'domaine/finalized_model.pkl'
    pickle.dump(model, open(filename, 'wb'))




def add_data(wine):
    """Add a wine to the data base

    Args:
        wine (WineModel): the wine we want to add
    """
    wine.to_csv("datasource/Wines.csv", mode='a', index=False, header=False)

def retrain_model():
    """Retrain the model with the current data
    """
    df=pd.read_csv("datasource/Wines.csv")
    df=df.drop(["Id"],axis=1)
    X = df.drop(['quality'], axis = 1)
    y = df['quality']
    # Normalize feature variables
    X_features = X
    X = StandardScaler().fit_transform(X)
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=0)
    model = RandomForestClassifier(random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    os.environ["Error"]=str(rnd_MSE)
    os.environ["Accuracy"]=str(rnd_score)
    dotenv.set_key("domaine/.env", "Error", os.environ["Error"])
    dotenv.set_key("domaine/.env", "Accuracy", os.environ["Accuracy"])
    filename = 'domaine/finalized_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    
def get_model_info():
    """Get current model info
    """
    load_model=pickle.load(open('domaine/finalized_model.pkl','rb'))
    
    return {
        "Parameter":load_model.get_params(),
        "Accuracy":os.environ.get("Accuracy"),
        "Error":os.environ.get("Error"),

    }

