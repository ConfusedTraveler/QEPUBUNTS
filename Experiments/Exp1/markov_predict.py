
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import sys
import os
from scipy.optimize import minimize_scalar,fsolve
import warnings

warnings.filterwarnings("ignore")

# Function to train a Markov model and estimate the transition matrix
def train_markov_model(train_data, N):
    # Initialize and populate a count matrix
    count_matrix = np.zeros((N, N))
    
    for i in range(len(train_data) - 1):
        current_state = train_data[i]
        next_state = train_data[i + 1]
        count_matrix[int(current_state)][int(next_state)] += 1
    
    # Normalize the count matrix to obtain the transition matrix
    transition_matrix = count_matrix / count_matrix.sum(axis=1, keepdims=True)
    
    return transition_matrix

# Function to evaluate a Markov model using the transition matrix
def markov_model(eval_data, transition_matrix):
    N = len(transition_matrix)
    predictions = []
    
    for i in range(len(eval_data) - 1):
        current_state = eval_data[i]
        
        
        # Predict the next state using the transition matrix
        predicted_next_state = np.argmax(transition_matrix[int(current_state)])
        predictions.append(predicted_next_state)
    
    return np.array(predictions)

def predict(time_series, N, out_put):

	# Split the time series into training (80%) and evaluation (20%) sets
	n = len(time_series)
	train_size = int(0.8 * n)
	train_data, eval_data = time_series[:train_size], time_series[train_size:]
	
	# Train the Markov model and estimate the transition matrix
	transition_matrix = train_markov_model(train_data, N)
	
	predictions = markov_model(eval_data, transition_matrix)
	
	with open(out_put,"a") as f:
		for j in range(len(predictions)):
			f.write(str(N)+","+str(predictions[j])+","+str(eval_data[j+1])+"\n")
			
			
	return
	
def get_data(path, length):
    df = pd.read_csv(path)
    df = df.iloc[:length, :]
    return df.iloc[:, 0].values 

	
if __name__ == "__main__":

    N_values = [5,10,15,20,25,30,35,40,45,50]
    n = 10000
    a_values = [1,15]
    # Get the current directory of the script
    current_directory = os.getcwd()
    
    # Navigate to the parent directory
    parent_directory = os.path.dirname(current_directory)
    
    # Navigate to the parent of the parent directory (grandparent directory)
    grandparent_directory = os.path.dirname(parent_directory)
    for a in a_values:
         for N in N_values:
            input_path = grandparent_directory + "/Datasets/Markov/Exp1/markov_ts_"+str(N)+"_"+str(50000)+"_"+str(a)+".csv"
            output_path = "Results/markov_pred_"+str(N)+"_"+str(n)+"_"+str(a)+".csv"
            series = get_data(input_path, n)
            predict(series,N,output_path)
	
	
			
			
