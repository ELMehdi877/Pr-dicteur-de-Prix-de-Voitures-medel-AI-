# 🚗 Prédicteur de Prix de Voitures d'Occasion

Ce projet de Machine Learning vise à estimer le prix de vente de véhicules d'occasion sur le marché indien à partir de leurs caractéristiques (année, kilométrage, marque, carburant, transmission, etc.). Il intègre un pipeline complet d'analyse, de nettoyage, de modélisation et une interface web interactive développée avec Streamlit.

---

## 📌 Table des Matières
- [Aperçu de la Solution](#-aperçu-de-la-solution)
- [Méthodologie & Choix Techniques](#-méthodologie--choix-techniques)
  - [1. Nettoyage & Préparation des Données](#1-nettoyage--préparation-des-données)
  - [2. Sélection & Optimisation du Modèle](#2-sélection--optimisation-du-modèle)
  - [3. Métriques d'Évaluation](#3-métriques-dévaluation)
- [Structure du Projet](#-structure-du-projet)
- [Installation & Configuration](#-installation--configuration)
- [Exécution de l'Application](#-exécution-de-lapplication)

---

## 🎯 Aperçu de la Solution

L'application permet à un utilisateur de saisir les caractéristiques clés d'un véhicule via une interface intuitive pour obtenir instantanément une estimation précise du prix calculée par le meilleur modèle pré-entraîné (`Random Forest` optimisé).

---

## 🛠 Méthodologie & Choix Techniques

### 1. Nettoyage & Préparation des Données
- **Traitement des valeurs manquantes & doublons :** Identification et suppression des enregistrements incohérents.
- **Encodage des variables catégorielles :** Application du *One-Hot Encoding* (`pd.get_dummies`) sur les variables telles que la marque (`brand`), le carburant (`fuel`), le type de vendeur (`seller_type`), la transmission (`transmission`) et le nombre de propriétaires (`owner`).
- **Séparation des données :** Découpage en ensembles d'entraînement (`X_train`, `y_train`) et de test (`X_test`, `y_test`) avec une répartition 80/20.

### 2. Sélection & Optimisation du Modèle
Plusieurs algorithmes de régression ont été entraînés et comparés sur les mêmes données :
- **Régression Linéaire** (Baseline rapide)
- **Support Vector Regression (SVR)**
- **XGBoost Regressor**
- **Random Forest Regressor** (Sélectionné comme meilleur modèle)

Le modèle **Random Forest** a ensuite été optimisé via **Recherche par Grille avec Validation Croisée (`GridSearchCV`, cv=5)**. 

### 3. Métriques d'Évaluation
Les modèles ont été comparés à l'aide de trois métriques principales :
- **$R^2$ (Coefficient de détermination) :** Détermine la capacité du modèle à expliquer la variance des prix ($R^2 = 0.7745$ pour le modèle optimisé).
- **MAE (Mean Absolute Error) :** Mesure l'erreur moyenne en valeur absolue ($\sim 123\,462 \text{ ₹}$).
- **RMSE (Root Mean Squared Error) :** Évalue la pénalisation des grandes erreurs ($\sim 207\,755 \text{ ₹}$).

---

## 📂 Structure du Projet

```text
├── data/
│   ├── extraction/      # Données brutes (car-price.csv)
│   ├── nettoyage/       # Données nettoyées (data_clean.csv)
│   └── prepare_date/    # Ensembles de train/test (X_train, X_test, etc.)
├── models/
│   └── best_random_forest.joblib  # Modèle final sérialisé
├── scripts/
│   ├── analyse_data.py  # Analyse exploratoire
│   ├── nettoyage_data.py# Nettoyage des données
│   ├── prepare_data.py  # Prétraitement et encodage
│   └── train_model.py    # Entraînement et GridSearchCV
├── app.py               # Interface web Streamlit
├── requirements.txt     # Dépendances Python
└── README.md            # Documentation du projet

# 🛠️ Installation et Exécution

## 1. Prérequis
Assurez-vous d'avoir **Python 3.8+** installé sur votre machine.

---

## 2. Cloner le dépôt et installer les dépendances

Lancer l'installation des bibliothèques nécessaires :
```bash
py -m pip install -r requirements.txt
```

---

## 3. Exécuter le pipeline de données (Optionnel)

Pour ré-exécuter le pipeline complet depuis le nettoyage jusqu'à l'entraînement :
```bash
python scripts/nettoyage_data.py
python scripts/prepare_data.py
python scripts/train_model.py
```

---

## 4. Lancer l'application Web Streamlit

Pour démarrer l'interface utilisateur interactive :
```bash
python -m streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse : [http://localhost:8501](http://localhost:8501).

---

# 🔧 Technologies Utilisées

- **Langage :** Python
- **Traitement de données :** Pandas, NumPy
- **Machine Learning :** Scikit-Learn, XGBoost, Joblib
- **Déploiement UI :** Streamlit