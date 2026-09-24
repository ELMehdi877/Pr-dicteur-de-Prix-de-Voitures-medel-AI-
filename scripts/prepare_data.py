import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_data = os.path.join(Base_path, "data/nettoyage/data_clean.csv")

df = pd.read_csv(path_data)

# ==============================================================================
# ÉTAPE 1 : EXTRACTION DE LA MARQUE & SÉPARATION (X, y)
# ==============================================================================

# On extrait la marque depuis la colonne 'name'
df["brand"] = df["name"].apply(lambda x: str(x).split()[0])

# Définition de la variable cible (y)
y = df["selling_price"]

# Définition des caractéristiques (X)
# On supprime 'selling_price' (la cible) ET 'name' (remplacé par 'brand')
X = df.drop(columns=["selling_price", "name"])

# Vérification des colonnes conservées dans X
# print("Colonnes dans X :", X.columns.tolist)


# ==============================================================================
# ÉTAPE 2 : SÉPARATION EN JEU D'ENTRAÎNEMENT ET DE TEST (80% / 20%)
# ==============================================================================

# Séparation des données : 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Vérification des dimensions des ensembles créés
print(f"Nombre d'échantillons d'entraînement (X_train) : {len(X_train)}")
print(f"Nombre d'échantillons de test (X_test) : {len(X_test)}")

# ==============================================================================
# ÉTAPE 3 : ENCODAGE DES VARIABLES CATÉGORIELLES
# ==============================================================================

# Encoding Ordinal pour 'owner' (respect de la hiérarchie)
owner_mapping = {
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
    "Test Drive Car": 0,
}

X_train["owner"] = X_train["owner"].map(owner_mapping).fillna(0)
X_test["owner"] = X_test["owner"].map(owner_mapping).fillna(0)

# One-Hot Encoding pour les autres colonnes catégorielles
categorical_cols = ["fuel", "seller_type", "transmission", "brand"]

# Transformation de X_train et X_test en colonnes 0/1
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Alignement des colonnes (pour que X_train et X_test aient exactement les mêmes colonnes)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

# Vérification rapide
print(f"Nombre de colonnes après encodage : \n X_train : {X_train.shape[1]} \n X_test : {X_test.shape[1]}")


# ==============================================================================
# ÉTAPE 4 : MISE À L'ÉCHELLE DES CARACTÉRISTIQUES (FEATURE SCALING)
# ==============================================================================

# Identification des colonnes numériques à mettre à l'échelle
num_cols = ["year", "km_driven", "owner"]

# Initialisation du Scaler
scaler = StandardScaler()

# Application du Scaler sur X_train (fit + transform)
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

# Application du Scaler sur X_test (transform uniquement !)
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Vérification rapide
print("Aperçu des premières lignes de X_train après Scaling :")
print(X_train[num_cols].head())


# Création du dossier pour stocker les données prêtes pour le ML
output_dir = "data/prepare_date"
os.makedirs(output_dir, exist_ok=True)

# Sauvegarde des 4 ensembles
X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

print("Tous les ensembles (Train/Test) ont été sauvegardés avec succès !")