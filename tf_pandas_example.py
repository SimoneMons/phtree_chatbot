import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# load the data frame
file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
dataframe = pd.read_csv(file_url)

# Separate features X from target y
X = dataframe.iloc[0:, 0:4] #data, features
y = dataframe.iloc[0:, 13:14] #target


# Create Train and test data to fit the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

'''
X_train_model = X_train.to_numpy()
y_train_model = y_train.to_numpy()

X_test_model = X_test.to_numpy()
y_test_model = y_test.to_numpy()
'''

'''
# scale
scaler = sk.MinMaxScaler(feature_range=(0, 250))
scaler = scaler.fit(X)
X_scaled = scaler.transform(X)
# Checking reconstruction
X_rec = scaler.inverse_transform(X_scaled)
'''

# scale X_train
Xscaler = MinMaxScaler(feature_range=(0, 1)) # scale so that all the X data will range from 0 to 1
Xscaler.fit(X_train)
scaled_X_train = Xscaler.transform(X_train)
print(scaled_X_train)

# scale y_train
Yscaler = MinMaxScaler(feature_range=(0, 1))
Yscaler.fit(y_train)
scaled_y_train = Yscaler.transform(y_train)
print(scaled_y_train.shape)
scaled_y_train = scaled_y_train.reshape(-1) # remove the second dimention from y so the shape changes from (n,1) to (n,)
print(scaled_y_train.shape)

# scale X_test
scaled_X_test = Xscaler.transform(X_test)


# design the neural network model
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))

model.add(Dense(1, activation='sigmoid'))

# define the loss function and optimization algorithm
model.compile(loss='mse', optimizer='adam')

# ft the model on the training dataset
model.fit(scaled_X_train, scaled_y_train, epochs=500, batch_size=10, verbose=0)



# make predictions for the input data
y_pred_scaled = model.predict(scaled_X_test)
#y_pred = Yscaler.inverse_transform(y_pred_scaled)

print(y_pred_scaled, y_test)

