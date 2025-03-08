import numpy as np
from scipy.stats import zipf
import sys
import os
import csv

def generate_transition_matrix(N, a):
    """
    Generate an N x N transition matrix using a Zipf distribution.
    
    Parameters:
        N (int): Number of states (i.e., size of the transition matrix).
        a (float): Parameter for the Zipf distribution. Must be greater than 1.
        
    Returns:
        normalized_matrix (np.ndarray): An N x N matrix where each row is normalized
                                          to sum to 1, representing transition probabilities.
    """
    # Ensure 'a' is greater than 1 for a valid Zipf distribution
    if a <= 1:
        raise ValueError("Parameter 'a' should be greater than 1 for Zipf distribution.")

    # Create a Zipf distribution object with parameter 'a'
    zipf_dist = zipf(a)
    # Generate an N x N array of random values based on the Zipf distribution
    values = zipf_dist.rvs(size=(N, N))
    # Normalize the generated values by dividing by the maximum value (scales the values to [0, 1])
    normalized_values = values / np.max(values)
    # Normalize each row so that the sum of probabilities in each row equals 1
    normalized_matrix = normalized_values / normalized_values.sum(axis=1, keepdims=True)

    return normalized_matrix


def generate_markov_time_series(n, N, alpha):
    """
    Generate a Markov time series and its corresponding transition matrix.
    
    Parameters:
        n (int): Total length of the time series.
        N (int): Number of states.
        alpha (float): Zipf distribution parameter used to generate the transition matrix.
    
    Returns:
        time_series (list): A list containing the generated time series states.
        transition_matrix (np.ndarray): The generated transition matrix.
    """
    # Initialize the time series with a random starting state between 0 and N-1
    time_series = [np.random.randint(N)]
    
    # Generate the transition matrix using the provided Zipf parameter (alpha)
    transition_matrix = generate_transition_matrix(N, alpha)
    
    # Generate the remaining states of the time series using the transition matrix probabilities
    for _ in range(n - 1):
        current_state = time_series[-1]
        # Use the current state's probabilities to select the next state
        next_state = np.random.choice(N, p=transition_matrix[current_state])
        time_series.append(next_state)
    
    return time_series, transition_matrix


if __name__ == "__main__":
    # Total number of states in the time series
    n = 50000
    # List of different N values (number of states) for which data will be generated
    N_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # Experiment identifiers; these may correspond to different experimental setups or conditions
    Exp = [1, 2]

    # Set seed value for reproducibility across experiments
    seed_value = 11
    # Specify the distribution used; here it is 'zipf'
    dist = "zipf"
    
    # Get the current directory of the script
    current_directory = os.getcwd()
    # Navigate to the parent directory; necessary for locating the Datasets folder
    parent_directory = os.path.dirname(current_directory)
    
    # Iterate over each experiment
    for x in Exp:
        Experiment = "Exp" + str(x)
        # Set alpha values depending on the experiment type
        if x == 1:
            alpha_values = [1.4, 15]
        else:
            alpha_values = [1.02, 15]
        
        # Loop through each number of states defined in N_values
        for N in N_values:
            # Loop through each alpha value to vary the transition probabilities
            for alpha in alpha_values:
                # Set the seed to ensure reproducible random sequences for each configuration
                np.random.seed(seed_value)
                
                # Generate the time series and transition matrix for the current configuration
                time_series, transition_matrix = generate_markov_time_series(n, N, alpha)
                
                # Construct the file path for saving the time series data (CSV format)
                ts_file = parent_directory + '/Datasets/Markov/' + Experiment + '/markov_ts_' + str(N) + '_' + str(n) + '_' + str(int(alpha)) + '.csv'
                # Construct the file path for saving the transition matrix data (CSV format)
                tr_file = parent_directory + '/Datasets/Markov/' + Experiment + '/markov_tr_' + str(N) + '_' + str(n) + '_' + str(int(alpha)) + '.csv'
                
                # Write the time series data to a CSV file; each row contains one state of the series
                with open(ts_file, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    for number in time_series:
                        csv_writer.writerow([number])
                
                # Write the transition matrix to a CSV file; each row represents the transition probabilities from a state
                with open(tr_file, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    for row in transition_matrix:
                        csv_writer.writerow(row)