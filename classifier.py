import pandas as pd

train_data = pd.read_csv("data/train.csv")

target = "Survived"

x_train = train_data.drop(target, axis=1)
y_train = train_data[target]

print(x_train, y_train)