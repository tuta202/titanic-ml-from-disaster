import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from ydata_profiling import ProfileReport
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np


data = pd.read_csv("data/train.csv")
x_test = pd.read_csv("data/test.csv", index_col="PassengerId")

# profile = ProfileReport(data, title="Titanic", explorative=True)
# profile.to_file("titanic.html")

target = "Survived"

data.set_index("PassengerId", inplace=True)

family_size_labels = ["Single", "Small", "Medium", "Large"]
data["Family_Size"] = data["SibSp"] + data["Parch"] + 1
data["Family_Size"] = pd.cut(data["Family_Size"], bins=[0,1,4,6,20], labels=family_size_labels)
x_test["Family_Size"] = x_test["SibSp"] + x_test["Parch"] + 1
x_test["Family_Size"] = pd.cut(x_test["Family_Size"], bins=[0,1,4,6,20], labels=family_size_labels)

def filter_name(name):
  match = re.search(r"\b([A-Z][a-z]+)\b(?=\.)", name)
  return match.group() if match else name

def group_title(title):
  if title in ["Mr", "Miss", "Mrs", "Master"]:
    return title
  elif title == "Ms":
    return "Miss"
  else:
    return "Others"
  
data["Title"] = data["Name"].apply(filter_name).apply(group_title)
x_test["Title"] = x_test["Name"].apply(filter_name).apply(group_title)

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
pclass_values = x_train["Pclass"].unique()
ord_transformer = Pipeline(steps=[
  ("imputer", SimpleImputer(strategy="most_frequent")),
  ("encoder", OrdinalEncoder(categories=[gender_values, pclass_values, family_size_labels])),
])

preprocessor = ColumnTransformer(transformers=[
  ("num_feature", num_transformer, ["Age", "Fare"]),
  ("nom_feature", nom_transformer, ["Cabin", "Embarked", "Ticket", "Title"]),
  ("ord_feature", ord_transformer, ["Sex", "Pclass", "Family_Size"]),
])

cls = Pipeline(steps=[
  ("preprocessor", preprocessor),
  # ("model", SVC()),
  ("model", RandomForestClassifier()),
])

params = {
  "model__n_estimators": [100, 200, 300],
  "model__criterion": ["gini", "entropy", "log_loss"],
  "model__max_depth": [None, 2, 5],
}

grid_search = GridSearchCV(estimator=cls, param_grid=params, cv=4, scoring="accuracy", verbose=2, n_jobs=-1)
grid_search.fit(x_train, y_train)
y_valid_predicted = grid_search.predict(x_valid)

print(classification_report(y_valid, y_valid_predicted))
print(grid_search.best_score_)
print(grid_search.best_params_)

y_test_predicted = grid_search.predict(x_test)

submission = pd.DataFrame({
  "PassengerId": x_test.index,
  "Survived": y_test_predicted
})
submission.to_csv("data/gender_submission.csv", index=False)