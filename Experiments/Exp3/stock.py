import numpy as np
import pandas as pd
import sys
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
#import lstm_predict
#import arima_predict
#import cnn_predict
from LZ1 import *
from LZ2 import *

def get_data(path, length):
    df = pd.read_csv(path)
    df = df.iloc[:length, :]
    return df.iloc[:, 0].values 
    



def get_pimax(S, N):
    #func = lambda x: (-(x * np.log2(x) + (1 - x) * np.log2(1 - x)) + (1 - x) * (np.log2(N - 1) - np.log2(1 - x))) - S
    #func = lambda x: -2*x * np.log2(x) - 2*(1 - x) * np.log2(1 - x) + (x) * (np.log2(N - 2)) - S
    func = lambda x: -x * np.log2(x) - (1 - x) * np.log2(1 - x) + (1 - x) * (np.log2(N - 2)) - S
    result = fsolve(func, 0.99999)
    return result[0]

    
if __name__ == "__main__":
	# Set the common parameters for AR model
	np.random.seed(0)
	model = sys.argv[1]
	#n = int(sys.argv[2])
	input_path = "Datasets/Stock_Open.csv"
	df = pd.read_csv(input_path,header=None, names=["Open"])
	series = df["Open"].values
	n = len(series)
	
	if model == "arima":
		output_path = "Stock_Results/arima_stock.csv"
		arima_predict.arima([n],series,output_path)
	elif model == "lstm":
		output_path = "Stock_Results/lstm_stock.csv"
		lstm_predict.train_lstm_and_save_predictions(series,filename=output_path)
	elif model == "cnnlstm":
		output_path = "Stock_Results/cnnlstm_stock.csv"
		cnn_predict.train_cnnlstm_and_save_predictions(series,filename=output_path)
	elif model== "Pimax":
		estimator = sys.argv[2]
		output_path = "Stock_Results/pimax_stock_"+estimator+".csv"
		epsilon = [0.1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60]
		pimax = []
		Hest = []
		if estimator == "Hcn":
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
		if estimator == "Hcn":
			with open(output_path,"a") as f:
				for j in range(len(epsilon)):
					f.write( str(n)+","+str(epsilon[j])+","+str(pimax[j])+","+str(Hest[j])+"\n")
		else:
			with open(output_path,"a") as f:
				for j in range(len(epsilon)):
					f.write( str(n)+","+str(epsilon[j])+","+str(pimax[j])+","+str(Hest[j])+"\n")
					
					
					
					
					
					
					
					
		
	
	
		
	
	
	

	
