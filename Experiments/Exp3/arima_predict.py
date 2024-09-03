import statsmodels.api as sm
from pmdarima import auto_arima
import numpy as np
import pandas as pd
import sys 

def arima(length,series,out_put,d=2):
	for i in range(len(length)):
		n = len(series)
		tr_size = int(np.ceil(0.8*n))
		te_size = int(n - tr_size)
		train = series[-n:-n+tr_size]
		testE = series[-n+tr_size:]
		predictions = np.around(predict_arima(train,testE),decimals=d)		
		with open(out_put,"a") as f:
			for j in range(len(predictions)):
				f.write(str(n)+","+str(predictions[j])+","+str(testE[j])+"\n")

def predict_arima(train,testE):
        
	stepwise_fit = auto_arima(train, 
		            suppress_warnings=True)
	order = stepwise_fit.get_params()['order']
	seasonal_order = stepwise_fit.get_params()['seasonal_order']
	predictions = []
	history = list(train)
	for t in range(len(testE)):
		model = sm.tsa.SARIMAX(history, order=order,seasonal_order=seasonal_order,initialization='approximate_diffuse')
		model_fit = model.fit()
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = testE[t]
		history.append(obs)

	return np.array(predictions)
