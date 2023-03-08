import numpy as np
import pandas as pd
from keras import layers
from keras import models
import tensorflow as tf
import tensorboard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from matplotlib import pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

num_epochs = 10

# encoding error with degrees symbol in utf-8
df = pd.read_csv("./SeoulBikeData.csv",
                 encoding_errors="ignore")
rename_dict = {}
rename_dict["Temperature(C)"] = "Temperature"
rename_dict["Humidity(%)"] = "Humidity"
rename_dict["Wind speed (m/s)"] = "Wind speed"
rename_dict["Visibility (10m)"] = "Visibility"
rename_dict["Dew point temperature(C)"] = "Dew point temperature"
rename_dict["Solar Radiation (MJ/m2)"] = "Solar Radiation"
rename_dict["Rainfall(mm)"] = "Rainfall"
rename_dict["Snowfall (cm)"] = "Snowfall"
df.rename(columns=rename_dict, inplace=True)


# Predict Rented Bike Count
# Predictors Hour, Temp, Hum, Solar Rad, Rain, Snow, Seasons, Holiday
# Filter Functional Day (Times where bike sharing was unavailible)

# print(df.columns)
# print(df.describe())
# print(df["Functioning Day"].describe())
# print(df.shape)

# df_EDA = df.loc[df["Functioning Day"] != "Yes", :]
# print(df_EDA)

# First filter valid rows
df_filtered = df.loc[df["Functioning Day"] == "Yes", :]

X = df_filtered.iloc[:, 2:12]
y = df_filtered.loc[:, "Rented Bike Count"]

# Encode values in X
X = pd.get_dummies(X, drop_first=True)
X = pd.get_dummies(X, columns=["Hour"], drop_first=True)
print(X.columns)
# print(X.describe())


# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=22)

# validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=22)

std = StandardScaler()
ct = ColumnTransformer([
    ('std', StandardScaler(), list(rename_dict.values()))
], remainder='passthrough')
ct.fit_transform(X_train, y_test)
ct.transform(X_val)
ct.transform(X_test)
print(X_train.head())
print(X_train.describe())

# print(X_train.shape)
# print(y_train.shape)
# print(X_val.shape)
# print(y_val.shape)
# print(X_test.shape)
# print(y_test.shape)

# Create the model


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(6, input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(0.2, seed=22))
    model.add(layers.Dense(3, activation='relu'))
    model.add(layers.Dropout(0.2, seed=22))
    model.add(layers.Dense(1))
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
model = build_model()
history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(
    X_test, y_test), batch_size=1, verbose=True, callbacks=[tensorboard_callback])
model.evaluate(X_test, y_test)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
