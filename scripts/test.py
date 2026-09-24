import pandas as pd
import os

Base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_data = os.path.join(Base_path, "data/nettoyage/data_clean.csv")

df = pd.read_csv(path_data)

owner = df["owner"]

owner = owner.drop_duplicates()

print(owner)