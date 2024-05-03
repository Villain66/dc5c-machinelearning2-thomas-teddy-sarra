import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le jeu de données
data = pd.read_csv("C:/Users/tvill/Documents/Clone/dc5c-machinelearning2-thomas-teddy-sarra/Datas.csv")

# Afficher les statistiques descriptives
print(data.describe())

# Pairplot pour visualiser les relations entre les variables
sns.pairplot(data)
plt.show()

# Exclure les variables catégorielles du calcul de la corrélation
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Calculer la matrice de corrélation sur les variables numériques uniquement
correlation_matrix = numerical_data.corr()

# Afficher la heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()