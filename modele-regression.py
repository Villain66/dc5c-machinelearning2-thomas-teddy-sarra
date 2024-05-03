import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt

# Charger le jeu de données
data = pd.read_csv("C:/Users/tvill/Documents/Clone/dc5c-machinelearning2-thomas-teddy-sarra/Datas.csv")

# Convertir toutes les variables catégorielles en variables binaires
data_encoded = pd.get_dummies(data)

# Diviser les données en variables indépendantes (X) et dépendante (y)
X = data_encoded.drop("Consommation énergétique(kWh)", axis=1)
y = data_encoded["Consommation énergétique(kWh)"]

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=46)

# Initialiser et entraîner le modèle de régression Ridge avec les meilleurs hyperparamètres
best_ridge_model = Ridge(alpha=10)
best_ridge_model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = best_ridge_model.predict(X_test)

# Calculer l'erreur quadratique moyenne (RMSE)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
