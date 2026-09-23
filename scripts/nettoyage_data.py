import pandas as pd
import os

# cree la path pour recuperer data
Base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_data = os.path.join(Base_path, "data/extraction/car-price.csv")

# cree le folder et le fichie sur lequel en dois mets data clean
data_clean = "data/netoyage"
os.makedirs(data_clean, exist_ok= True)

# recuperation de data de puis le fichie csv
df = pd.read_csv(path_data)

# data frame
df = pd.DataFrame(df)

# supprimer les lines doublons
df = df.drop_duplicates()

# remplacer les valeurs monquant de year par le median
df["year"] = df.groupby("name")["year"].transform(lambda x: x.fillna(x.median()))
df["year"] = df["year"].fillna(df["year"].median())
df["year"] = df["year"].round().astype(int)


# remplacer les valeurs monquant de km_driven par le median
df["km_driven"] = df.groupby(["year", "owner"])["km_driven"].transform(lambda x: x.fillna(x.median()))
df["km_driven"] = df.groupby("year")["km_driven"].transform(lambda x: x.fillna(x.median()))
df["km_driven"] = df["km_driven"].fillna(df["km_driven"].median())


# remplacer les valeurs monquant de fuel avec le mode
df["fuel"] = df.groupby("name")["fuel"].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None)
    )
df["fuel"] = df["fuel"].fillna(df["fuel"].mode()[0])

# remplacer les valeurs monquant de seller_type avec le mode
df["price_group"] = pd.qcut(df["selling_price"], q=4, labels=["pas cher", "moyen", "elever", "luxe"])
df["seller_type"] = df.groupby(["price_group", "year"])["seller_type"].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None)
    )

df.drop(columns=["price_group"], inplace=True)

# remplacer les valeurs monquant de owner avec le mode
df["km_group"] = pd.qcut(df["km_driven"], q=4, labels=["tres peu", "moyen", "elever", "tres elever"])
df["owner"] = df.groupby(["year", "km_group"])["owner"].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None)
    )

df.drop(columns=["km_group"], inplace=True)

path_clean = os.path.join(data_clean, "data_clean.csv")

df.to_csv(path_clean, index=False)