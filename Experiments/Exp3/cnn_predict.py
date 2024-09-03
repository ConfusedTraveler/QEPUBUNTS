import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler


def train_cnnlstm_and_save_predictions(time_series_data, n_steps=10, n_splits=5, epochs=50, batch_size=32, filename="lstm_predictions.csv"):
	# Create a DataFrame
	df = pd.DataFrame(time_series_data, columns=['Value'])

	# Split data into training and test sets
	train_size = int(len(df) * 0.8)  # 80% training, 20% test
	train_data, test_data = df[:train_size], df[train_size:]



	# Initialize Min-Max scaler and scale training data

	# Split data into input (X) and output (y) sequences
	def create_sequences(data, n_steps):
		X, y = [], []
		for i in range(len(data) - n_steps):
			X.append(data[i:i + n_steps])
			y.append(data[i + n_steps])
		return np.array(X), np.array(y)

	# Split training data for cross-validation
	X_train, y_train = create_sequences(train_data['Value'].values, n_steps)
	# Initialize Min-Max scaler and scale data
	scaler = MinMaxScaler()
	y_scaler = MinMaxScaler()
	X_train = scaler.fit_transform(X_train)
	y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))

	# Initialize CNN-LSTM model
	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, 1)))
	model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
	model.add(LSTM(50, activation='relu'))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')

	# Initialize Time Series Split cross-validator
	tscv = TimeSeriesSplit(n_splits=n_splits)

	# Perform cross-validation on training data
	all_predictions = []
	all_actual_values = []

	for train_index, val_index in tscv.split(X_train):
		X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
		y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

		model.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size, verbose=0)

		y_pred = model.predict(X_val_fold)
		y_pred = y_scaler.inverse_transform(y_pred)
		y_val_fold = y_scaler.inverse_transform(y_val_fold)
		all_predictions.extend(y_pred)
		all_actual_values.extend(y_val_fold)


	all_predictions = []
	all_actual_values = []
	# Evaluate the model on the test set
	X_test, y_test = create_sequences(test_data['Value'].values, n_steps)
	X_test = scaler.transform(X_test)
	y_test = y_scaler.transform(y_test.reshape(-1, 1))

	y_pred_test = model.predict(X_test)
	y_pred_test = y_scaler.inverse_transform(y_pred_test)
	y_test = y_scaler.inverse_transform(y_test)

	all_predictions.extend(y_pred_test.reshape(-1,))
	all_actual_values.extend(y_test.reshape(-1,))

	l = np.ones(len(all_actual_values))*len(time_series_data)
	# Save predictions and actual values to a CSV file
	predictions_df = pd.DataFrame({'Actual': all_actual_values, 'Predicted': all_predictions})
	predictions_df.to_csv(filename, index=False)

	return predictions_df

