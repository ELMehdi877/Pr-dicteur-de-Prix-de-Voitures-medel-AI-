import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==============================================================================
# 1. INITIALISATION DES CHEMINS ET CHARGEMENT DU JEU DE DONNÉES
# ==============================================================================

# Construction du chemin absolu dynamique vers le fichier source CSV
Base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_data = os.path.join(Base_path, "data/extraction/car-price.csv")

# Récupération des données brutes depuis le fichier CSV
df = pd.read_csv(path_data)
df = pd.DataFrame(df)

# ==============================================================================
# 2. INSPECTION STATISTIQUE INITIALE ET DIAGNOSTIC DES DONNÉES
# ==============================================================================

# Affichage des dimensions du DataFrame (Nombre de lignes, Nombre de colonnes)
print(df.shape)

# Diagnostic de la structure : Types de données (int, float, object) et présence de valeurs non-nulles
df.info()

# Résumé des statistiques descriptives pour les colonnes numériques (moyenne, écart-type, quartiles, min/max)
print(df.describe())

# Comptage du nombre de valeurs manquantes (NaN) par colonne
print(df.isnull().sum())

# ==============================================================================
# 3. VISUALISATION DES ANOMALIES & PREMIÈRE IMPUTATION DE 'YEAR'
# ==============================================================================

# Extraction du premier mot de la colonne 'name' pour obtenir le nom de la marque
df["Brand"] = df["name"].apply(lambda x: str(x).split()[0])

# Visualisation par Boxplot : Distribution des années de mise en circulation par marque
plt.figure(figsize=(12, 6))
sns.boxplot(x="Brand", y="year", data=df)
plt.xticks(rotation=90)
plt.show()

# Imputation temporaire des valeurs manquantes de 'year' par la médiane du modèle ('name') puis la médiane globale
df["year"] = df.groupby("name")["year"].transform(
    lambda x: x.fillna(x.median())
)
df["year"] = df["year"].fillna(df["year"].median())

# ==============================================================================
# 4. ANALYSE DES RELATIONS ET DES PROPRIÉTAIRES ('KM_DRIVEN' VS 'YEAR')
# ==============================================================================

# --- Graphique 1 : Nuage de points (Scatter Plot) ---
# Objectif : Observer la corrélation entre le kilométrage et l'année, segmentée par type de propriétaire
plt.figure(figsize=(10, 6))

sns.scatterplot(data=df, x="year", y="km_driven", hue="owner", alpha=0.7)

plt.title("Relation entre l'Année, le Kilométrage et le Nombre de Propriétaires")
plt.xlabel("Année de mise en circulation")
plt.ylabel("Kilométrage (km_driven)")
plt.legend(title="Propriétaire")
plt.grid(True)
plt.show()

# --- Graphique 2 : Diagramme en barres (Count Plot) ---
# Objectif : Analyser la distribution des catégories 'owner' selon l'âge des véhicules
plt.figure(figsize=(12, 6))

sns.countplot(data=df, x="year", hue="owner")

plt.title(
    "Répartition du type de propriétaire (Owner) selon l'année de mise en"
    " circulation"
)
plt.xlabel("Année (Year)")
plt.ylabel("Nombre de véhicules")
plt.xticks(rotation=45)
plt.legend(title="Owner")
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()

# ==============================================================================
# 5. DÉTECTION GLOBALE DES VALEURS ABERRANTES (BOXPLOTS BIVARIÉS)
# ==============================================================================

# Visualisation côte à côte de la distribution des variables numériques principales
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.boxplot(y=df["selling_price"], ax=axes[0]).set_title("Prix de vente")
sns.boxplot(y=df["km_driven"], ax=axes[1]).set_title("Kilométrage")
sns.boxplot(y=df["year"], ax=axes[2]).set_title("Année")

plt.tight_layout()
plt.show()