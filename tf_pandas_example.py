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

#val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
#train_dataframe = dataframe.drop(val_dataframe.index)

'''
print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

print(val_dataframe)
'''

print(X_train)

X_train_model = X_train.to_numpy()
y_train_model = y_train.to_numpy()

#print(X_train_model)

X_test_model = X_test.to_numpy()



'''
#print(y_train)

Yscaler = MinMaxScaler(feature_range=(0, 1))
Yscaler.fit(y_train)
scaled_y_train = Yscaler.transform(y_train)
#print(scaled_y_train)
scaled_y_train = scaled_y_train.reshape(-1) # remove the second dimention from y so the shape changes from (n,1) to (n,)



Xscaler = MinMaxScaler(feature_range=(0, 1)) # scale so that all the X data will range from 0 to 1
Xscaler.fit(X_train)
scaled_X_train = Xscaler.transform(X_train)

scaled_y_train = np.insert(scaled_y_train, 0, 0)
scaled_y_train = np.delete(scaled_y_train, -1)
print(scaled_X_train)
print(scaled_y_train)

features = ['age', 'sex', 'cp', 'trestbps']

x_training_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(X_train[features].values, tf.float32),
        )
    )
)

print(x_training_dataset)
'''

# design the neural network model
model = Sequential()
model.add(Dense(9, input_dim=4, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(9, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))

# define the loss function and optimization algorithm
model.compile(loss='mse', optimizer='adam')

# ft the model on the training dataset
model.fit(X_train_model, y_train_model, epochs=500, batch_size=10, verbose=0)



# make predictions for the input data
yhat = model.predict(X_test_model)

print(yhat)

print(len(yhat))

print(y_test)
