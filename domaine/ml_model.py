import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#https://moncoachdata.com/blog/apprendre-machine-learning-avec-python/

wines = pd.read_csv("Wines.csv")
wines.corr()["quality"]
# alcohol ~ 0.5, bonne corrélation
# sulphates ~ 0.26
# citric acid ~0.24
# fixed acidity 0.12
columns = wines.columns.tolist()
columns = [c for c in columns if c not in ["quality"]]  

target = "quality"
#wines[wines["quality"] == 8].iloc[0]

# Importer une fonction prévue pour séparer les sets.
from sklearn.model_selection import train_test_split
# Générer le set de training. Fixer random_state pour répliquer les resultats ultérieurement.
train = wines.sample(frac=0.8, random_state=1)
# Sélectionner tout ce qui n'est pas dans le set de training et le mettre dans le set de test.
test = wines.loc[~wines.index.isin(train.index)]

# Importer la fonction de calcul d'erreur depuis scikit-learn.
from sklearn.metrics import mean_squared_error


# Importer le modèle random forest.
from sklearn.ensemble import RandomForestRegressor
# Initialiser le modèle avec certains paramètres.
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
# Adapter le modèle aux données.
model.fit(train[columns], train[target])
# Faire des prédictions.
predictions = model.predict(test[columns])
# Calculer l'erreur.
print(mean_squared_error(predictions, test[target]))
