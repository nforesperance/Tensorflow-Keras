# example of calculation 1d convolutions
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv1D
# define input data
data = asarray([0, 0, 0, 1, 1, 0, 0, 0])
data = data.reshape(1, 8, 1) # (samples,lenght,channels)

# create model
model = Sequential()
#We will define a model that expects input samples to have the shape [lenght, chanels].
# Conv1D(number_samples, lenth(weight), input_shape=(lenght(input), channels)
model.add(Conv1D(1, 3, input_shape=(8, 1)))


# define a vertical line detector
weights = [asarray([[[0]],[[1]],[[0]]]), asarray([0.0])] #[0,1,0]
# store the weights in the model
model.set_weights(weights)
# confirm they were stored
print(model.get_weights())
# apply filter to input data
yhat = model.predict(data)
print(yhat)