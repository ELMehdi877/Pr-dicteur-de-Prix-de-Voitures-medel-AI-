import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

Base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_data = os.path.join(Base_path, "data/extraction/car-price.csv")

# recuperer le data de puis le fichie csv
df = pd.read_csv(path_data)
df = pd.DataFrame(df)

# voir la taille du fichie
# print(df.shape)

# # voir le type de chaque colone
# df.info()

# # voir les calcules statistique
# print(df.describe())

# # voir les valeurs null
# print(df.isnull().sum())


# graphique de year en de fonctions  marque
# df["Brand"] = df["name"].apply(lambda x: str(x).split()[0])
# plt.figure(figsize=(12, 6))
# sns.boxplot(x="Brand", y="year", data=df)
# plt.xticks(rotation=90)
# plt.show()

# df["year"] = df.groupby("name")["year"].transform(lambda x: x.fillna(x.median()))
# df["year"] = df["year"].fillna(df["year"].median())

# graphique de km_driven en fpnction de year

# plt.figure(figsize=(10, 6))

# # Nuage de points : km_driven vs year avec couleur selon owner
# sns.scatterplot(
#     data=df, 
#     x="year", 
#     y="km_driven", 
#     hue="owner", 
#     alpha=0.7
# )

# plt.title("Relation entre l'Année, le Kilométrage et le Nombre de Propriétaires")
# plt.xlabel("Année de mise en circulation")
# plt.ylabel("Kilométrage (km_driven)")
# plt.legend(title="Propriétaire")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(12, 6))

# # Diagramme de comptage : Année sur l'axe X, couleur selon le nombre de propriétaires
# sns.countplot(data=df, x="year", hue="owner")

# plt.title("Répartition du type de propriétaire (Owner) selon l'année de mise en circulation")
# plt.xlabel("Année (Year)")
# plt.ylabel("Nombre de véhicules")
# plt.xticks(rotation=45)
# plt.legend(title="Owner")
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.boxplot(y=df["selling_price"], ax=axes[0]).set_title("Prix de vente")
sns.boxplot(y=df["km_driven"], ax=axes[1]).set_title("Kilométrage")
sns.boxplot(y=df["year"], ax=axes[2]).set_title("Année")
plt.tight_layout()
plt.show()
