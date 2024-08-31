import pandas as pd
import numpy as np
import concurrent.futures
import sys
sys.path.append('../../') 

from Utils.LZ2 import * 
from Utils.LZ1 import *
from Utils.compute_PIMax import *
from Utils.computeH_exact import *
import warnings
np.random.seed(0)
warnings.filterwarnings("ignore")


def generate_transition_matrix(N,p):
    P = np.zeros((N, N))

    for i in range(N):
        # Set the dominant next state probability
        dominant_prob = 1 - (p*(N-1))  # You can adjust this value based on your requirements

        # Choose a random dominant next state not equal to the current state
        possible_next_states = [j for j in range(N) if j != i]
        dominant_next_state = np.random.choice(possible_next_states)

        # Set the dominant next state probability
        P[i, dominant_next_state] = dominant_prob

        # Set equiprobable probabilities for other states
        equiprobable_prob = p #(1 - dominant_prob) / (N - 1)
        for j in range(N):
            if j != dominant_next_state:
                P[i, j] = equiprobable_prob

    
    return P

def generate_sequence(transition_matrix, initial_state, sequence_length):
    current_state = initial_state
    sequence = [current_state]
    
    for _ in range(sequence_length - 1):
        next_state = np.random.choice(len(transition_matrix[current_state]), p=transition_matrix[current_state])
        sequence.append(next_state)
        current_state = next_state
    
    return sequence

def read_csv_values(csv_file_path,n):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path, header=None)

    # Assuming the CSV file has a single column, return the values as a list
    return df.iloc[:n, 0].values

def read_tr_pvalues(csv_file_path):

	df = pd.read_csv(csv_file_path, header=None) 
	#print(df.head())
	
	return df   
def process_file(values,tr_p, N,p):
	
	print("Started for p=",p)

	with concurrent.futures.ThreadPoolExecutor() as executor:
		# Submit each function to the executor with parameters
		H_LZ1 = executor.submit(Compute_LZ1, values, 0)
		H_LZ2 = executor.submit(Compute_LZ2, values, 0)
		H_exact = executor.submit(exact_H,values, N, tr_p,[p,1-p])

		# Wait for all futures to complete
		concurrent.futures.wait([H_LZ1, H_LZ2, H_exact])


	HLZ1 = H_LZ1.result()
	HLZ2 = H_LZ2.result()
	Hexact = H_exact.result()
	
	return N,p,HLZ1,HLZ2,Hexact
	
def main():
    
    output_file_name = './Results/exp2_fig2_10000_10.txt'
    futures = []
    N = 10
    n = 10000
    p_tr = [1/(N*10**i) for i in range(8)]
    print(p_tr)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for p in p_tr:
            # Generate transition probability matrix
            transition_matrix = generate_transition_matrix(N,p)
            # Generate a sequence

            initial_state = np.random.choice(N)
            sequence = generate_sequence(transition_matrix, initial_state, n)
            res = executor.submit(process_file,sequence,transition_matrix, N,p)
            futures.append(res)

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

        # Retrieve the results
        results = [future.result() for future in futures]	

    with open(output_file_name,'a') as of:
        of.write(f'N,p,HLZ1,HLZ2,Hexact\n')
        for res in results:
            N,p,HLZ1,HLZ2,Hexact = res
            of.write(f'{N},{p},{HLZ1},{HLZ2},{Hexact}\n')
    	
if __name__ == "__main__":
    main()







