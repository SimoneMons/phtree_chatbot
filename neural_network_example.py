# example of creating a univariate dataset with a given mapping function
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import math


x = np.arange(-10, 10, 0.1)
print(x)

# define the input data
#x = [i for i in range(-50, 51)]
#print(x)
# define the output data
#y = [math.sin(i) for i in x]

y = np.sin(x) + np.cos(x)


# plot the input versus the output

'''
pyplot.scatter(x, y)
pyplot.title('Input (x) versus Output (y)')
pyplot.xlabel('Input Variable (x)')
pyplot.ylabel('Output Variable (y)')
pyplot.show()
'''


# define the dataset
#x = np.asarray([i for i in range(-50,51)])
x = np.arange(-10, 10, 0.1)
x = x.reshape((len(x), 1))
#y = np.asarray([i**2.0 for i in x])
y = np.sin(x) + np.cos(x)
y = y.reshape((len(y), 1))



scale_x = MinMaxScaler()
x = scale_x.fit_transform(x)
scale_y = MinMaxScaler()
y = scale_y.fit_transform(y)
#print(x.min(), x.max(), y.min(), y.max())


# design the neural network model
model = Sequential()
model.add(Dense(4, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(9, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(9, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(9, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(9, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))

# define the loss function and optimization algorithm
model.compile(loss='mse', optimizer='Adam')

# ft the model on the training dataset
model.fit(x, y, epochs=500, batch_size=10, verbose=0)

# make predictions for the input data
yhat = model.predict(x)

#print(yhat)

x_plot = scale_x.inverse_transform(x)
y_plot = scale_y.inverse_transform(y)
yhat_plot = scale_y.inverse_transform(yhat)

pyplot.scatter(x_plot, yhat_plot)
pyplot.title('Input (x) versus Output (y)')
pyplot.xlabel('Input Variable (x)')
pyplot.ylabel('Output Variable (y)')
pyplot.show()




