import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib

# ==============================================================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES PRÉPARÉES
# ==============================================================================

# Définition du chemin vers le dossier des données prétraitées
Base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_prepare_date = os.path.join(Base_path, "data/prepare_date")

# Chargement des 4 ensembles (X_train, X_test, y_train, y_test)
X_train = pd.read_csv(os.path.join(path_prepare_date, "X_train.csv"))
X_test = pd.read_csv(os.path.join(path_prepare_date, "X_test.csv"))
y_train = pd.read_csv(os.path.join(path_prepare_date, "y_train.csv"))
y_test = pd.read_csv(os.path.join(path_prepare_date, "y_test.csv"))

# Vérification du chargement
print("Données chargées avec succès !")
print(f"Dimensions de X_train : {X_train.shape}")
print(f"Dimensions de X_test  : {X_test.shape}")


# ==============================================================================
# ÉTAPE 2 : INITIALISATION DES 4 MODÈLES ET DES MÉTRIQUES
# ==============================================================================

# Dictionnaire contenant les 4 modèles demandés par le cahier des charges
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "SVR": SVR(),
}

print("Les 4 modèles ont été initialisés avec succès :")
for model_name in models.keys():
    print(f"- {model_name}")


# ==============================================================================
# ÉTAPE 3 : ENTRAÎNEMENT ET PRÉDICTIONS
# ==============================================================================

# Dictionnaire pour stocker les prédictions de chaque modèle
predictions = {}

# Boucle d'entraînement sur les 4 modèles
for name, model in models.items():
    print(f"Entraînement du modèle : {name}...")
    
    # 1. Entraînement du modèle
    model.fit(X_train, y_train.values.ravel())
    
    # 2. Prédiction sur l'ensemble de test
    predictions[name] = model.predict(X_test)

print("\nTous les modèles ont été entraînés avec succès !")


# ==============================================================================
# ÉTAPE 4 : CALCUL DES MÉTRIQUES ET TABLEAU DE COMPARAISON
# ==============================================================================

results = []

# Calcul des métriques pour chaque modèle
for name, y_pred in predictions.items():
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append(
        {
            "Modèle": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R²": round(r2, 4),
        }
    )

# Conversion des résultats en DataFrame pour un affichage propre
df_results = pd.DataFrame(results)

# Tri des modèles du meilleur au moins bon (selon le R²)
df_results = df_results.sort_values(by="R²", ascending=False)

print("\n================ COMPARAISON DES PERFORMANCES ================")
print(df_results.to_string(index=False))


# ==============================================================================
# ÉTAPE 5 : DÉFINITION DE LA GRILLE D'HYPERPARAMÈTRES
# ==============================================================================

# 1. Sélection du meilleur modèle de la phase précédente
rf = RandomForestRegressor(random_state=42)

# 2. Définition de la grille de paramètres à tester
param_grid = {
    # "n_estimators": [100, 200, 300],
    # "max_depth": [10, 20, None],
    # "min_samples_split": [2, 5, 10],
    # "min_samples_leaf": [1, 2, 4],
    # "n_estimators": [200, 300],
    # "max_depth": [15, 20, 25, None],
    # "min_samples_split": [2, 5],
    # "min_samples_leaf": [2, 4],
    "n_estimators": [100, 200, 300],
    "max_depth": [15, 25, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

print("Grille d'hyperparamètres définie avec succès !")



# ==============================================================================
# ÉTAPE 6 : EXÉCUTION DE GRIDSEARCHCV AVEC VALIDATION CROISÉE
# ==============================================================================

# 1. Initialisation de GridSearchCV (cv=5 pour 5-fold cross-validation)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,  # Utilise tous les cœurs du processeur pour accélérer
    verbose=1,
)

print("Lancement de la recherche par grille (GridSearchCV)...")

# 2. Entraînement sur les données d'apprentissage (X_train, y_train)
grid_search.fit(X_train, y_train.values.ravel())

# 3. Affichage des meilleurs paramètres trouvés
print("\nRecherche terminée avec succès !")
print(f"Meilleurs hyperparamètres : {grid_search.best_params_}")
print(
    f"Meilleur score R² en Validation Croisée : {round(grid_search.best_score_, 4)}"
)


# Récupération du meilleur modèle issu de la recherche
best_rf = grid_search.best_estimator_

# Prédiction sur l'ensemble de test réel
y_pred_best = best_rf.predict(X_test)

# Calcul des métriques finales sur X_test
mae_opt = mean_absolute_error(y_test, y_pred_best)
rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_opt = r2_score(y_test, y_pred_best)

print("\n================ RÉSULTATS DU MODÈLE OPTIMISÉ SUR X_TEST ================")
print(f"MAE  : {round(mae_opt, 2)}")
print(f"RMSE : {round(rmse_opt, 2)}")
print(f"R²   : {round(r2_opt, 4)}")



# ==============================================================================
# ÉTAPE 7 : TABLEAU COMPARATIF FINAL (BASELINE VS OPTIMISÉ)
# ==============================================================================

# Calcul du R² final pour le modèle optimisé sur X_test
r2_opt = r2_score(y_test, y_pred_best)

# Ajout du modèle optimisé aux résultats
results.append(
    {
        "Modèle": "Random Forest (Optimisé)",
        "MAE": round(mae_opt, 2),
        "RMSE": round(rmse_opt, 2),
        "R²": round(r2_opt, 4),
    }
)

# Conversion et affichage du tableau final
df_final = pd.DataFrame(results).sort_values(by="R²", ascending=False)

print("\n================ COMPARAISON FINALE DES MODÈLES ================")
print(df_final.to_string(index=False))



# ==============================================================================
# ÉTAPE 8 : ANALYSE DE L'IMPORTANCE DES VARIABLES
# ==============================================================================

# 1. Extraction des importances depuis le meilleur modèle
importances = best_rf.feature_importances_

# 2. Création d'un DataFrame pour lier chaque variable à son importance
df_importance = pd.DataFrame(
    {"Variable": X_train.columns, "Importance": importances}
)

# 3. Tri des variables de la plus importante à la moins importante
df_importance = df_importance.sort_values(by="Importance", ascending=False)

# 4. Affichage des 10 variables les plus influentes
print("\n================ TOP 10 DES VARIABLES LES PLUS IMPORTANTES ================")
print(df_importance.head(10).to_string(index=False))



# ==============================================================================
# ÉTAPE 9 : SAUVEGARDE DU MODÈLE OPTIMISÉ (.joblib)
# ==============================================================================

# 1. Création du dossier models/ s'il n'existe pas encore
models_dir = os.path.join(Base_path, "models")
os.makedirs(models_dir, exist_ok=True)

# 2. Exportation du meilleur modèle
model_path = os.path.join(models_dir, "best_random_forest.joblib")
joblib.dump(best_rf, model_path)

print(
    f"\nLe modèle optimisé a été sauvegardé avec succès dans :"
    f" {model_path}"
)
