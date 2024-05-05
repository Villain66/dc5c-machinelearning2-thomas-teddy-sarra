import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Charger le jeu de données
data = pd.read_csv("C:/Users/tvill/Documents/Clone/dc5c-machinelearning2-thomas-teddy-sarra/Datas.csv")

# Convertir toutes les variables catégorielles en variables binaires
data_encoded = pd.get_dummies(data)

# Diviser les données en variables indépendantes (X) et dépendante (y)
X = data_encoded.drop("Consommation énergétique(kWh)", axis=1)
y = data_encoded["Consommation énergétique(kWh)"]

# Diviser les données en ensemble d'entraînement et de validation/test
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.5, random_state=44)

# Initialiser et entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de validation/test
y_pred = model.predict(X_validation)

# Calculer le coefficient de détermination (R²)
r2 = r2_score(y_validation, y_pred)
print("Coefficient de détermination (R²) :", r2)

# Obtenir les coefficients de régression
coefficients = model.coef_

# Associer les coefficients aux variables correspondantes
variable_coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': coefficients})

# Trier les coefficients par valeur absolue pour mettre en évidence les plus importants
variable_coefficients = variable_coefficients.reindex(variable_coefficients['Coefficient'].abs().sort_values(ascending=False).index)

# Afficher les coefficients
print(variable_coefficients)

# Visualisation des coefficients
plt.figure(figsize=(10, 6))
plt.barh(variable_coefficients['Variable'], variable_coefficients['Coefficient'])
plt.xlabel('Coefficient de régression')
plt.title('Importance des variables dans la prédiction de la consommation énergétique')
plt.show()

# Créer le scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_validation, y_pred, color='blue', label='Prédictions')
plt.plot([min(y_validation), max(y_validation)], [min(y_validation), max(y_validation)], color='red', linestyle='--', label='Réel')
plt.title('Comparaison entre les valeurs prédites et réelles')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.legend()
plt.grid(True)
plt.show()