import numpy as np
from scipy.stats import zipf
import sys
import csv


def generate_zipf_transition_matrix(N, a):
    # Ensure a is greater than 1 for Zipf distribution
    if a <= 1:
        raise ValueError("Parameter 'a' should be greater than 1 for Zipf distribution.")

    # Create a Zipf distribution object
    zipf_dist = zipf(a)

    # Generate random variates from the Zipf distribution
    values = zipf_dist.rvs(size=(N, N))

    # Normalize values to be between 0 and 1
    normalized_values = values / np.max(values)

    # Normalize each row to ensure the sum is 1
    normalized_matrix = normalized_values / normalized_values.sum(axis=1, keepdims=True)

    return normalized_matrix
    
    
def generate_markov_time_series(n, N, alpha, dist):
    # Initialize the time series with a random integer between 0 and N-1
    time_series = [np.random.randint(N)]
    if dist == "zipf":
        #alpha = 3.0  # Shape parameter for the Zipf distribution

        # Generate the transition matrix using a Zipf distribution
        transition_matrix = generate_zipf_transition_matrix(N, alpha)
    else: 
        # Generate a random transition matrix
        transition_matrix = np.random.rand(N, N)
        transition_matrix /= transition_matrix.sum(axis=1)[:, np.newaxis]  # Normalize rows to make it a valid transition matrix
    
    # Generate the rest of the time series
    for _ in range(n - 1):
        current_state = time_series[-1]
        next_state = np.random.choice(N, p=transition_matrix[current_state])
        time_series.append(next_state)

    return time_series, transition_matrix
    
    
if __name__ == "__main__":

	n = 50000
	N_values = [2,3,4,5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
	
	alpha_values = [1.02, 2, 15]
	seed_values = [11, 41, 71, 307, 739]
	
	
	dist = "zipf"
	uz = "z"
	for seed_value in seed_values:
		for N in N_values:
			for alpha in alpha_values:
	  

				# Set the seed for NumPy's random number generator
				np.random.seed(seed_value)
				
				time_series, transition_matrix = generate_markov_time_series(n, N, alpha, dist)
				
				# Name of the CSV file
				ts_file = './Seed'+str(seed_value)+'/markov_ts_' + str(N)+'_'+str(n)+'_'+str(int(alpha))+uz+'.csv'
				
				# Name of the CSV file
				tr_file = './Seed'+str(seed_value)+'/markov_tr_' + str(N)+'_'+str(n)+'_'+str(int(alpha))+uz+'.csv'
				
				

				# Open the CSV file in write mode
				with open(ts_file, 'w', newline='') as csvfile:
					# Create a CSV writer
					csv_writer = csv.writer(csvfile)

					# Write each number as a separate row
					for number in time_series:
						csv_writer.writerow([number])
						
					# Open the CSV file in write mode
				with open(tr_file, 'w', newline='') as csvfile:
					# Create a CSV writer
					csv_writer = csv.writer(csvfile)

					# Write each number as a separate row
					for row in transition_matrix:
						csv_writer.writerow(row)
	
	
