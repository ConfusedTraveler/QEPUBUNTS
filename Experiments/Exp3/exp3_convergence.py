import numpy as np
import pandas as pd
import sys
import csv
from scipy.optimize import fsolve
from LZ2 import *


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

	if len(sys.argv) <= 3:
		raise ValueError("Usage: python3 script.py <dataset> <length> <model>")

	dataset = sys.argv[1]
	n = int(sys.argv[2])
	model = sys.argv[3]

	# Assign tolerance values based on the argument
	if dataset == "temperature":
		tolerance_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
		input_path = "Datasets/temperature.csv"
		series = get_data(input_path, n)
	elif dataset == "stock":
		tolerance_values = [0.01, 0.1, 1, 5, 10,20,30,40,50,60]
		input_path = "Datasets/Stock_Open.csv"
		series = get_data(input_path, n)
	elif dataset == "ETTh1":
		tolerance_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
		input_path = "Datasets/ETT-data/ETTh1.csv"
		df = pd.read_csv(input_path)
		df = df.iloc[:n, :]
		series = df["OT"].values
	elif dataset == "markov":
		tolerance_values = [float(sys.argv[4])]
		N = int(sys.argv[5])
		alpha = int(sys.argv[6])
		input_path = "Datasets/Markov/Seed11/markov_ts_"+str(N)+"_"+str(50000)+"_"+str(alpha)+sys.argv[7]+".csv"
		output_path = "Results/Pimax/Markov/"+dataset+sys.argv[2]+"_"+str(N)+"_"+str(alpha)+sys.argv[7]+"ZZ.csv"
		series = get_data(input_path, n)
	elif dataset == "AR1":
		tolerance_values = [0.005, 0.01, 0.02]
		input_path = "Datasets/AR1.csv"
		series = get_data(input_path, n)
		
	else:
		raise ValueError("Dataset should be one of these: <temperature> <stock> <markov3> <markov10> <markov3> <markov10> <markov3> <markov10>")

	
	if model == "arima":
		output_path = "Results/Arima/"+dataset+sys.argv[2]+".csv"
		arima_predict.arima([n],series,output_path)
	elif model == "lstm":
		output_path = "Results/LSTM/"+dataset+sys.argv[2]+".csv"
		lstm_predict.train_lstm_and_save_predictions(series,filename=output_path)
	elif model == "cnnlstm":
		output_path = "Results/CNNLSTM/"+dataset+sys.argv[2]+".csv"
		cnn_predict.train_cnnlstm_and_save_predictions(series,output_path)
		
	elif model == "markov":
		output_path = "Results/Markov/" + dataset + '_'+sys.argv[2]+"_"+str(N)+".csv"
		markov_predict.predict(series,N,output_path)
	elif model == "Pimax":
		epsilon = tolerance_values
		pimax = []
		Hest = []
		for e in epsilon:
			N = (max(series)-min(series)+2*e)/e
			H = Compute_LZ2(series,e)
			pimax.append(get_pimax(H,N))
			Hest.append(H)
		with open(output_path,"a") as f:
			for j in range(len(epsilon)):
				f.write( str(n)+","+str(epsilon[j])+","+str(pimax[j])+","+str(Hest[j])+"\n")
					
					
					
					
					
					
					
					
		
	
	
		
	
	
	

	
