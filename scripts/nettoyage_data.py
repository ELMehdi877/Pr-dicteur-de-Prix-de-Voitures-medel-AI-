import pandas as pd
import os

# ==============================================================================
# 1. INITIALISATION DES CHEMINS ET CHARGEMENT DES DONNÉES
# ==============================================================================

# Création du chemin absolu pour récupérer le dataset brut
Base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_data = os.path.join(Base_path, "data/extraction/car-price.csv")

# Création du dossier de destination pour les données nettoyées
data_clean = "data/nettoyage"
os.makedirs(data_clean, exist_ok=True)

# Chargement du fichier CSV dans le DataFrame
df = pd.read_csv(path_data)
df = pd.DataFrame(df)

# ==============================================================================
# 2. NETTOYAGE DE BASE & IMPUTATION DES VALEURS MANQUANTES
# ==============================================================================

# Suppression des doublons stricts dans le jeu de données
df = df.drop_duplicates()

# --- Imputation de 'year' ---
# Remplacement des valeurs manquantes par la médiane par modèle ('name'), puis globale
df["year"] = df.groupby("name")["year"].transform(lambda x: x.fillna(x.median()))
df["year"] = df["year"].fillna(df["year"].median())
df["year"] = df["year"].round().astype(int)

# --- Imputation de 'km_driven' ---
# Imputation conditionnelle par (Année, Propriétaire), puis par Année, puis globale
df["km_driven"] = df.groupby(["year", "owner"])["km_driven"].transform(lambda x: x.fillna(x.median()))
df["km_driven"] = df.groupby("year")["km_driven"].transform(lambda x: x.fillna(x.median()))
df["km_driven"] = df["km_driven"].fillna(df["km_driven"].median())

# --- Imputation de 'fuel' ---
# Imputation par le Mode (valeur la plus fréquente) selon le modèle de voiture
df["fuel"] = df.groupby("name")["fuel"].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None)
)
df["fuel"] = df["fuel"].fillna(df["fuel"].mode()[0])

# --- Imputation de 'seller_type' ---
# Regroupement par tranche de prix et année pour deviner le type de vendeur
df["price_group"] = pd.qcut(df["selling_price"], q=4, labels=["pas cher", "moyen", "elever", "luxe"])
df["seller_type"] = df.groupby(["price_group", "year"])["seller_type"].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None)
)
df.drop(columns=["price_group"], inplace=True)

# --- Imputation de 'owner' ---
# Regroupement par tranche de kilométrage et année pour deviner le nombre de propriétaires
df["km_group"] = pd.qcut(df["km_driven"], q=4, labels=["tres peu", "moyen", "elever", "tres elever"])
df["owner"] = df.groupby(["year", "km_group"])["owner"].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None)
)
df.drop(columns=["km_group"], inplace=True)

# ==============================================================================
# 3. TRAITEMENT DES VALEURS ABERRANTES (OUTLIERS)
# ==============================================================================

# --- A. Traitement de la variable 'year' ---
# Justification : Les véhicules d'avant 2000 sont très rares et n'obéissent pas aux lois du marché standard.
df = df[df["year"].between(2000, 2026)]

# --- B. Traitement de la variable 'km_driven' ---
# Calcul des bornes IQR dynamique groupe par groupe d'année (une moustache IQR par 'year')
Q1_km_driven = df.groupby("year")["km_driven"].transform(lambda x: x.quantile(0.25))
Q3_km_driven = df.groupby("year")["km_driven"].transform(lambda x: x.quantile(0.75))
IQR_km_driven = Q3_km_driven - Q1_km_driven
upper_bound_km_driven = Q3_km_driven + 1.5 * IQR_km_driven

# Plafonnement (Capping) des kilométrages extrêmes au lieu d'une suppression
df["km_driven"] = df["km_driven"].clip(upper=upper_bound_km_driven)

# --- C. Traitement de la variable 'selling_price' ---
# Conservation de la valeur brute pour la visualisation finale
df["selling_price_raw"] = df["selling_price"]

# 1. Extraction de la marque (premier mot du champ 'name')
df["brand"] = df["name"].apply(lambda x: str(x).split()[0])

# 2. Filtrage des erreurs absolues de prix (prix aberrants < 10 000 ou > 6 000 000)
df = df[df["selling_price"].between(10000, 6000000)]

# 3. Calcul de la borne IQR spécifique au groupe (Année + Marque)
Q1_price_group = df.groupby(["year", "brand"])["selling_price"].transform(lambda x: x.quantile(0.25))
Q3_price_group = df.groupby(["year", "brand"])["selling_price"].transform(lambda x: x.quantile(0.75))
IQR_price_group = Q3_price_group - Q1_price_group
upper_bound_price1 = Q3_price_group + 1.5 * IQR_price_group

# 4. Calcul de la borne IQR de secours (Fallback) par Année seule
Q1_price_year = df.groupby("year")["selling_price"].transform(lambda x: x.quantile(0.25))
Q3_price_year = df.groupby("year")["selling_price"].transform(lambda x: x.quantile(0.75))
IQR_price_year = Q3_price_year - Q1_price_year
upper_bound_price2 = Q3_price_year + 1.5 * IQR_price_year

# 5. Sélection de la borne finale : si IQR_price_group == 0 (marque rare), on prend la borne de l'année
upper_bound_price_final = upper_bound_price1.where(IQR_price_group > 0, upper_bound_price2)

# 6. Plafonnement dynamique des prix de vente
df["selling_price"] = df["selling_price"].clip(upper=upper_bound_price_final)

# ==============================================================================
# 4. VISUALISATION ET COMPARISON (AVANT / APRÈS)
# ==============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Graphique 1 : Avant Capping (Données brutes)
sns.scatterplot(
    data=df,
    x="year",
    y="selling_price_raw",
    hue="brand",
    legend=False,
    ax=axes[0],
)
axes[0].set_title("Prix AVANT Capping (Prix réels)")
axes[0].grid(True)

# Graphique 2 : Après Capping Intelligente
sns.scatterplot(
    data=df, x="year", y="selling_price", hue="brand", legend=False, ax=axes[1]
)
axes[1].set_title("Prix APRÈS Capping Intelligente (Year + Brand)")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Suppression des colonnes temporaires de travail
df.drop(columns=["selling_price_raw", "brand"], inplace=True)

# ==============================================================================
# 5. SAUVEGARDE DU DATASET NETTOYÉ
# ==============================================================================

path_clean = os.path.join(data_clean, "data_clean.csv")
df.to_csv(path_clean, index=False)