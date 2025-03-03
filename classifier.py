import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

data = pd.read_csv("data/train.csv")
x_test = pd.read_csv("data/test.csv", index_col="PassengerId")

target = "Survived"

data.set_index("PassengerId", inplace=True)
family_size_labels = ["Single", "Small", "Medium", "Large"]
data["Family_Size"] = data["SibSp"] + data["Parch"] + 1
data["Family_Size"] = pd.cut(data["Family_Size"], bins=[0,1,4,6,20], labels=family_size_labels)

x = data.drop(target, axis=1)
y = data[target]
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

num_transformer = Pipeline(steps=[
  ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
  ("scaler", StandardScaler()),
])

nom_transformer = Pipeline(steps=[
  ("imputer", SimpleImputer(strategy="most_frequent")),
  ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

gender_values = x_train["Sex"].unique()
ord_transformer = Pipeline(steps=[
  ("imputer", SimpleImputer(strategy="most_frequent")),
  ("encoder", OrdinalEncoder(categories=[gender_values, family_size_labels])),
])

preprocessor = ColumnTransformer(transformers=[
  ("num_feature", num_transformer, ["Age", "Fare"]),
  ("nom_feature", nom_transformer, ["Cabin", "Embarked", "Ticket"]),
  ("ord_feature", ord_transformer, ["Sex", "Family_Size"]),
])

x_train = preprocessor.fit_transform(x_train)

print(x_train)
