import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt


from scipy.optimize import minimize_scalar,fsolve
import warnings
import sys
sys.path.append('../')  # Add parent directory to the system path

from Utils.LZ2 import *
from Utils.LZ1 import *
from Utils.compute_PIMax import *


warnings.filterwarnings("ignore")

seed = 37

#Function to generate transition matrix
def generate_transition_matrix(N,p):

    P = np.zeros((N, N))
    np.random.seed(seed)

    for i in range(N):
        
        dominant_prob = 1 - (p*(N-1))  
        possible_next_states = [j for j in range(N) if j != i]
        dominant_next_state = np.random.choice(possible_next_states)
        P[i, dominant_next_state] = dominant_prob
        equiprobable_prob = p #(1 - dominant_prob) / (N - 1)
        
        for j in range(N):
            if j != dominant_next_state:
                P[i, j] = equiprobable_prob

    
    return P

# Function to train a Markov model and estimate the transition matrix
def train_markov_model(train_data, N):
    
    np.random.seed(seed)
    count_matrix = np.zeros((N, N))
    for i in range(len(train_data) - 1):
        current_state = train_data[i]
        next_state = train_data[i + 1]
        count_matrix[int(current_state)][int(next_state)] += 1
    
    transition_matrix = count_matrix / count_matrix.sum(axis=1, keepdims=True)
    
    return transition_matrix

# Function to evaluate a Markov model using the transition matrix
def markov_model(eval_data, transition_matrix):
    N = len(transition_matrix)
    np.random.seed(seed)
    predictions = []
    
    for i in range(len(eval_data) - 1):
        current_state = eval_data[i]
        predicted_next_state = np.argmax(transition_matrix[int(current_state)])
        predictions.append(predicted_next_state)
    
    return np.array(predictions)

def predict(time_series, N):
	np.random.seed(seed)
	n = len(time_series)
	train_size = int(0.8 * n)
	train_data, eval_data = time_series[:train_size], time_series[train_size:]
	transition_matrix = train_markov_model(train_data, N)
	predictions = markov_model(eval_data, transition_matrix)			
	return predictions,eval_data

def generate_sequence(transition_matrix, initial_state, sequence_length):
    current_state = initial_state
    sequence = [current_state]
    for _ in range(sequence_length - 1):
        next_state = np.random.choice(len(transition_matrix[current_state]), p=transition_matrix[current_state])
        sequence.append(next_state)
        current_state = next_state
    return sequence	

def get_acc(predictions,eval_data):
	p = 0
	for i in range(len(predictions)):
		if predictions[i]==eval_data[i+1]:
			p+=1		
	acc = p/len(predictions)			
	return acc			

def discretize(S,pred,eval_data,e):
	
	N = int((max(S) - min(S) + 2 * e) / (2 * e))
	bins = np.linspace(min(S) - e, max(S) + e, N + 1)
	S_discretized = pd.cut(S, bins=bins, labels=False)
	pred_discretized = pd.cut(pred, bins=bins, labels=False)
	eval_discretized = pd.cut(eval_data, bins=bins, labels=False)
	N = len(np.unique(S_discretized))
	return S_discretized,pred_discretized,eval_discretized,N
	
if __name__ == "__main__":

    first = int(sys.argv[1])

    
    np.random.seed(seed)


    output_path = "Results/24_50_fig2.csv"
    eps = [1,2.5,5,7.5,10,12.5,15,17.5,20]
    if first == 1:
        N = int(sys.argv[2])
        n = int(sys.argv[3])
        p = 1/(N*1.95)
        transition_matrix = generate_transition_matrix(N,p)
        
        initial_state = np.random.choice(N)
        series = generate_sequence(transition_matrix, initial_state, n)
        tsize = int(0.8*len(series))
        predictions, eval_data = predict(series,N)
        pimax_l = []
        pimarkov = []
        
        for e in eps:
            print("eps:",e)
            S,pred,edata,NN = discretize(series,predictions,eval_data,e)
            H = Compute_LZ2(S,0)
            pimax = get_pimax(H,NN)		
            acc = get_acc(pred,edata)
            pimax_l.append(pimax)
            pimarkov.append(acc)
        csv_file = f'Results/{initial_state}_{N}_fig2.csv'
        data = list(zip(pimax_l, pimarkov))

        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Pimax', 'Pimarkov'])
            writer.writerows(data)
    else:
         series = pd.read_csv(output_path)
         pimax_l = series['Pimax'].values
         pimarkov = series['Pimarkov'].values

    plt.figure(figsize=(8, 5))	
    plt.plot(eps,pimax_l,marker= "o",label="Pimax")
    plt.plot(eps,pimarkov,marker="o",label="Pimarkov")
    plt.ylim(0.3,0.8)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Bin Width',fontsize=20)
    plt.ylabel('Accuracy/Pimax',fontsize=20)
    plt.legend(fontsize=16,loc='lower right')
    plt.tight_layout()
    plt.savefig(f"./Results/fig2.png")
    plt.show()
    plt.close()		
