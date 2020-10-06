# example of creating a univariate dataset with a given mapping function
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot

# define the input data
x = [i for i in range(-50, 51)]
print(x)
# define the output data
y = [i ** 2.0 for i in x]

# plot the input versus the output
#pyplot.scatter(x, y)
#pyplot.title('Input (x) versus Output (y)')
#pyplot.xlabel('Input Variable (x)')
#pyplot.ylabel('Output Variable (y)')
#pyplot.show()

# define the dataset
x = np.asarray([i for i in range(-50,51)])
x = x.reshape((len(x), 1))
y = np.asarray([i**2.0 for i in x])
y = y.reshape((len(y), 1))

scale_x = MinMaxScaler()
x = scale_x.fit_transform(x)
scale_y = MinMaxScaler()
y = scale_y.fit_transform(y)
print(x.min(), x.max(), y.min(), y.max())


# design the neural network model
model = Sequential()
model.add(Dense(9, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(9, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))

# define the loss function and optimization algorithm
model.compile(loss='mse', optimizer='adam')

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


# creating a deep learning model with keras
def build_model():
    model = Sequential()

    model.add(Dense(64, input_dim=4, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(Adam(lr=lr, decay=decay), loss='mse')
    model.summary()
    return model

model = build_model()

# running the game
for i_episodes in range(200):
    env.reset()
    for i in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        # observation = ndarray float64
        # reward = float
        # done = bool
        # action = int
        # info = empty

        observation = np.asarray(observation)
        reward = np.asarray(reward)
        action = np.asarray(action)

        model.fit(np.expand_dims(observation, axis=0), np.expand_dims(action, axis=0))