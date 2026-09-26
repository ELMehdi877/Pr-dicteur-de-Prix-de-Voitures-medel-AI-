import os
import joblib
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. CHARGEMENT DU MODÈLE ET DES COLONNES
# ==============================================================================
# Définition des chemins
Base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(Base_path, "..", "models", "best_random_forest.joblib")
data_path = os.path.join(Base_path, "..", "data", "prepare_date", "X_train.csv")

# Si le fichier est à la racine, ajuster les chemins simplement
if not os.path.exists(model_path):
    model_path = os.path.join(Base_path, "models", "best_random_forest.joblib")
    data_path = os.path.join(Base_path, "data", "prepare_date", "X_train.csv")

# Chargement du modèle et du fichier d'entraînement pour récupérer le nom des colonnes
model = joblib.load(model_path)
X_train = pd.read_csv(data_path)

# ==============================================================================
# 2. INTERFACE GRAPHIQUE STREAMLIT
# ==============================================================================
st.title("🚗 Estimation du Prix de Voitures d'Occasion")
st.write("Entrez les caractéristiques de votre véhicule pour estimer son prix.")

# Liste des marques disponibles
brands = [
    "Maruti",
    "Hyundai",
    "Mahindra",
    "Tata",
    "Ford",
    "Honda",
    "Toyota",
    "BMW",
    "Mercedes-Benz",
    "Audi",
]

# Champs de saisie (6 caractéristiques)
year = st.slider(
    "Année de fabrication", min_value=1990, max_value=2024, value=2018
)
km_driven = st.number_input(
    "Kilométrage (km)", min_value=0, max_value=500000, value=50000, step=1000
)
fuel = st.selectbox(
    "Type de carburant", ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
)
seller_type = st.selectbox(
    "Type de vendeur", ["Individual", "Dealer", "Trustmark Dealer"]
)
transmission = st.selectbox("Boîte de vitesses", ["Manual", "Automatic"])
owner = st.selectbox(
    "Nombre de propriétaires précédents",
    [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car",
    ],
)
brand = st.selectbox("Marque du véhicule", brands)

# ==============================================================================
# 3. PRÉDICTION LORS DU CLIC SUR LE BOUTON
# ==============================================================================
if st.button("💰 Estimer le prix"):
    # Création d'une ligne vide avec toutes les colonnes du modèle initialisées à 0
    input_data = pd.DataFrame(0, index=[0], columns=X_train.columns)

    # --- A. Transformation Ordinale de 'owner' (Conforme à prepare_data.py) ---
    owner_mapping = {
        "First Owner": 1,
        "Second Owner": 2,
        "Third Owner": 3,
        "Fourth & Above Owner": 4,
        "Test Drive Car": 0
    }
    owner_val = owner_mapping.get(owner, 0)

    # Remplissage des valeurs numériques
    input_data["year"] = year
    input_data["km_driven"] = km_driven
    input_data["owner"] = owner_val

    # Activation des colonnes spécifiques (One-Hot Encoding)
    if f"fuel_{fuel}" in input_data.columns:
        input_data[f"fuel_{fuel}"] = 1

    if f"seller_type_{seller_type}" in input_data.columns:
        input_data[f"seller_type_{seller_type}"] = 1

    if f"transmission_{transmission}" in input_data.columns:
        input_data[f"transmission_{transmission}"] = 1

    if f"brand_{brand}" in input_data.columns:
        input_data[f"brand_{brand}"] = 1

    # Calcul de la prédiction
    predicted_price = model.predict(input_data)[0]

    # Sécurité pour éviter les prix négatifs ou nuls sur les cas extrêmes
    prix_final = max(10000.0, predicted_price)

    # Affichage du résultat
    st.success(
        f"Le prix estimé pour ce véhicule est de : **{round(prix_final, 2):,} ₹**"
    )