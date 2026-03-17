import numpy as np
import pandas as pd
import sys
sys.path.append('../../') 

from Utils.LZ2 import * 
from Utils.LZ1 import *
from Utils.compute_PIMax import *
from scipy.optimize import fsolve
from tqdm import tqdm

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
	input_path = "../../Datasets/Stock_Open.csv"
	df = pd.read_csv(input_path,header=None, names=["Open"])
	series = df["Open"].values
	n = len(series)
	
	if model == "arima":
		output_path = "Stock_Results/arima_stock.csv"
		print(f"运行ARIMA模型预测，数据长度: {n}")
		arima_predict.arima([n],series,output_path)
		print("ARIMA预测完成")
	elif model == "lstm":
		output_path = "Stock_Results/lstm_stock.csv"
		print(f"运行LSTM模型训练和预测，数据长度: {n}")
		lstm_predict.train_lstm_and_save_predictions(series,filename=output_path)
		print("LSTM预测完成")
	elif model == "cnnlstm":
		output_path = "Stock_Results/cnnlstm_stock.csv"
		print(f"运行CNN-LSTM模型训练和预测，数据长度: {n}")
		cnn_predict.train_cnnlstm_and_save_predictions(series,filename=output_path)
		print("CNN-LSTM预测完成")
	elif model== "Pimax":
		estimator = sys.argv[2]
		output_path = "Stock_Results/pimax_stock_"+estimator+".csv"
		epsilon = [0.1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60]
		pimax = []
		Hest = []
		print(f"运行PIMax计算，估计器: {estimator}, epsilon数量: {len(epsilon)}")
		if estimator == "NLZ1":
			for e in tqdm(epsilon, desc="NLZ1计算进度"):
				N = (max(series)-min(series)+2*e)/e	
				H = Compute_LZ1(series,e)
				pimax.append(get_pimax(H,N))
				Hest.append(H)
		else:
			for e in tqdm(epsilon, desc="NLZ2计算进度"):
				N = (max(series)-min(series)+2*e)/e
				H = Compute_LZ2(series,e)
				pimax.append(get_pimax(H,N))
				Hest.append(H)
		print(f"保存结果到 {output_path}")
		if estimator == "NLZ1":
			with open(output_path,"a") as f:
				for j in range(len(epsilon)):
					f.write( str(n)+","+str(epsilon[j])+","+str(pimax[j])+","+str(Hest[j])+"\n")
		else:
			with open(output_path,"a") as f:
				for j in range(len(epsilon)):
					f.write( str(n)+","+str(epsilon[j])+","+str(pimax[j])+","+str(Hest[j])+"\n")
		print("PIMax计算完成")
					
					
					
					
					
					
					
					
		
	
	
		
	
	
	

	
