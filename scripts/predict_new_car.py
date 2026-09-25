# import os
# import joblib
# import pandas as pd

# # ==============================================================================
# # TEST DU MODÈLE SAUVEGARDÉ (.joblib)
# # ==============================================================================

# # 1. Définition des chemins
# Base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# model_path = os.path.join(Base_path, "models", "best_random_forest.joblib")
# path_prepare_date = os.path.join(Base_path, "data", "prepare_date")

# # 2. Chargement du modèle sauvegardé
# loaded_model = joblib.load(model_path)
# print("Modèle .joblib chargé avec succès !")

# # 3. Chargement de quelques données de test (ex: les 5 premières lignes)
# X_test = pd.read_csv(os.path.join(path_prepare_date, "X_test.csv"))
# y_test = pd.read_csv(os.path.join(path_prepare_date, "y_test.csv"))

# X_sample = X_test.head(30)
# y_real = y_test.head(30).values.ravel()

# # 4. Prédiction avec le modèle chargé
# y_pred = loaded_model.predict(X_sample)

# # 5. Affichage comparatif (Prix Réel vs Prix Prédit)
# df_compare = pd.DataFrame(
#     {
#         "Prix Réel": y_real,
#         "Prix Prédit": y_pred.round(2),
#         "Écart": (y_real - y_pred).round(2),
#     }
# )

# print("\n================ COMPARAISON SUR 5 VOITURES DE TEST ================")
# print(df_compare.to_string(index=False))

import os
import joblib
import pandas as pd

# 1. Chargement du modèle et d'un exemple de données
Base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(Base_path, "models", "best_random_forest.joblib")
path_prepare_date = os.path.join(Base_path, "data", "prepare_date")

model = joblib.load(model_path)
X_test = pd.read_csv(os.path.join(path_prepare_date, "X_test.csv"))

# 2. On prend la première ligne pour avoir TOUTES les colonnes
ma_voiture = X_test.iloc[[0]].copy()

# 3. Réinitialisation de toutes les colonnes à 0 sauf les données numériques
ma_voiture[:] = 0

# 4. Définition des caractéristiques de ta voiture
ma_voiture["year"] = 2018
ma_voiture["km_driven"] = 50000

# Choisir 1 pour les options correspondant à la voiture (0 pour le reste)
ma_voiture["fuel_Diesel"] = 1
ma_voiture["seller_type_Individual"] = 1
ma_voiture["transmission_Manual"] = 1
ma_voiture["owner"] = 1
ma_voiture["brand_Maruti"] = 1  # Change la marque selon ton choix

# 5. Prédiction du prix
prix_predit = model.predict(ma_voiture)[0]

print(f"Prix estimé pour cette voiture : {round(prix_predit, 2)} ₹")