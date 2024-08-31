import numpy as np
from scipy.stats import zipf
import sys
import os
import csv

def generate_transition_matrix(N, a):
    # Ensure a is greater than 1 for Zipf distribution
    if a <= 1:
        raise ValueError("Parameter 'a' should be greater than 1 for Zipf distribution.")

    zipf_dist = zipf(a)
    values = zipf_dist.rvs(size=(N, N))
    normalized_values = values / np.max(values)
    # Normalize each row to ensure the sum is 1
    normalized_matrix = normalized_values / normalized_values.sum(axis=1, keepdims=True)

    return normalized_matrix


def generate_markov_time_series(n, N, alpha):
    # Initialize the time series with a random integer between 0 and N-1
    time_series = [np.random.randint(N)]

    transition_matrix = generate_transition_matrix(N, alpha)

    # Generate the rest of the time series
    for _ in range(n - 1):
        current_state = time_series[-1]
        next_state = np.random.choice(N, p=transition_matrix[current_state])
        time_series.append(next_state)

    return time_series, transition_matrix

if __name__ == "__main__":

    n = 50000
    N_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    Exp = [1,2]

    
    seed_value = 11
    dist = "zipf"
    # Get the current directory of the script
    current_directory = os.getcwd()
    
    # Navigate to the parent directory
    parent_directory = os.path.dirname(current_directory)
    
    for x in Exp:
        Experiment ="Exp"+str(x)
        if x == 1:
            alpha_values = [1.4,15]
        else:
            alpha_values = [1.02,15]
        for N in N_values:
            for alpha in alpha_values:
                # Set the seed for NumPy's random number generator
                np.random.seed(seed_value)

                time_series, transition_matrix = generate_markov_time_series(n, N, alpha)

                # Name of the CSV file
                ts_file = parent_directory + '/Datasets/Markov/'+Experiment+'/markov_ts_' + str(N)+'_'+str(n)+'_'+str(int(alpha))+'.csv'

                # Name of the CSV file
                tr_file = parent_directory + '/Datasets/Markov/'+Experiment+'/markov_tr_' + str(N)+'_'+str(n)+'_'+str(int(alpha))+'.csv'

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