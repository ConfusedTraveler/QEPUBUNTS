import sys
import numpy as np
from Utils import *

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

np.random.seed(41)
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def get_model(Neurons, params ):
    n_steps = params[0]
    n_features = params[1]
    
    # define model
    model = Sequential()
    model.add(LSTM(Neurons[0], activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    for i in range(1,len(Neurons)-1):
        model.add(LSTM(Neurons[i], return_sequences= True, activation='relu'))
    model.add(LSTM(Neurons[-1],activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def predict(model,test):
    pred = []
    for i in range(len(test)):
        x_input = test[i]
        yhat = model.predict(x_input, verbose=0)
        pred.append(yhat[0][0])
    ypred = np.array(pred)
    return ypred

def train_lstm_and_save_predictions(series,filename):
	
	
	N = len(series)
	n_steps = 10
	n_features =1
	params = [n_steps,n_features]
	tr_size = int(0.8*N)
	train = series[-N:-N+tr_size]
	test = series[-N+tr_size-n_steps:]
	# choose a number of time steps
	# split into samples
	Neurons = [256,128,64,32]
	Xtrain, ytrain = split_sequence(train, n_steps)
	Xtest, ytest = split_sequence(test, n_steps)
	# reshape from [samples, timesteps] into [samples, timesteps, features]
	Xtrain = Xtrain.reshape((Xtrain.shape[0], n_steps, n_features))
	Xtest = Xtest.reshape((Xtest.shape[0], n_steps, n_features))
	model = get_model(Neurons,params)
	model.fit(Xtrain, ytrain, epochs=50, verbose=0)
	Ttest = []

	
	for j in range(len(ytest)):
		x_input = Xtest[j].reshape(1,n_steps,n_features)
		Ttest.append(x_input)
	predictions = predict(model,Ttest)     
	with open(filename,"a") as file:
		for pred,y in zip(predictions, ytest):
			file.write(f"{pred},{y}\n")
