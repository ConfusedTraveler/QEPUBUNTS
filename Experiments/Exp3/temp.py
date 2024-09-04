import numpy as np
import pandas as pd
import sys
sys.path.append('../../') 

from Utils.LZ2 import * 
from Utils.LZ1 import *
from Utils.compute_PIMax import *
from scipy.optimize import fsolve

import lstm_predict
import arima_predict
import cnn_predict


def get_data(path, length):
    df = pd.read_csv(path)
    df = df.iloc[:length, :]
    return df.iloc[:, 0].values 
    



def get_pimax(S, N):
    func = lambda x: -x * np.log2(x) - (1 - x) * np.log2(1 - x) + (1 - x) * (np.log2(N - 2)) - S
    result = fsolve(func, 0.99999)
    return result[0]

    
if __name__ == "__main__":
	# Set the common parameters for AR model
	np.random.seed(0)
	model = sys.argv[1]
	#n = int(sys.argv[2])
	input_path = "../../Datasets/temperature.csv"
	df = pd.read_csv(input_path)
	print(df.head())
	print(df.shape)
	series = df["HT"].values
	n = 16000 #len(series)
	series = series[:n]
	if model == "arima":
		output_path = "Temp_Results/arima_temp.csv"
		arima_predict.arima([n],series,output_path)
	elif model == "lstm":
		output_path = "Temp_Results/lstm_temp.csv"
		lstm_predict.train_lstm_and_save_predictions(series,filename=output_path)
	elif model == "cnnlstm":
		output_path = "Temp_Results/cnnlstm_temp.csv"
		cnn_predict.train_cnnlstm_and_save_predictions(series,filename=output_path)
	elif model== "Pimax":
		estimator = sys.argv[2]
		output_path = "Temp_Results/pimax_temp_"+estimator+".csv"
		epsilon = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]
		pimax = []
		Hest = []
		if estimator == "NLZ1":
			for e in epsilon:
				N = (max(series)-min(series)+2*e)/e	
				H = Compute_LZ1(series,e)
				pimax.append(get_pimax(H,N))
				Hest.append(H)
		else:
			for e in epsilon:
				N = (max(series)-min(series)+2*e)/e
				H = Compute_LZ2(series,e)
				pimax.append(get_pimax(H,N))
				Hest.append(H)
		if estimator == "NLZ1":
			with open(output_path,"a") as f:
				for j in range(len(epsilon)):
					f.write( str(n)+","+str(epsilon[j])+","+str(pimax[j])+","+str(Hest[j])+"\n")
		else:
			with open(output_path,"a") as f:
				for j in range(len(epsilon)):
					f.write( str(n)+","+str(epsilon[j])+","+str(pimax[j])+","+str(Hest[j])+"\n")
					
					
					
					
					
					
					
					
		
	
	
		
	
	
	

	
