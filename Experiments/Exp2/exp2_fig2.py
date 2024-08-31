import os
import pandas as pd
import sys
sys.path.append('../../') 

from Utils.LZ2 import * 
from Utils.LZ1 import *
from Utils.compute_PIMax import *
from Utils.computeH_exact import *

import concurrent.futures



def read_csv_values(csv_file_path,n):
	# Read the CSV file into a pandas DataFrame
	df = pd.read_csv(csv_file_path, header=None)

	# Assuming the CSV file has a single column, return the values as a list
	return df.iloc[:n, 0].values

def read_tr_pvalues(csv_file_path):

	df = pd.read_csv(csv_file_path, header=None)
	return df
   
def process_file(values, tr_p,N):

	with concurrent.futures.ThreadPoolExecutor() as executor:
		# Submit each function to the executor with parameters
		H_LZ1 = executor.submit(Compute_LZ1, values, 0.1)
		H_LZ2 = executor.submit(Compute_LZ2, values, 0.1)
		H_exact = executor.submit(get_exactH, N, tr_p)

		# Wait for all futures to complete
		concurrent.futures.wait([H_LZ1, H_LZ2, H_exact])


	HLZ1 = H_LZ1.result()
	HLZ2 = H_LZ2.result()
	Hexact = H_exact.result()

	return N,HLZ1,HLZ2,Hexact

        
def main():

	# List of integers for N
	N_values = [2,3,4,5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
	n = 10000
	dist = '15'


	# Get the current directory of the script
	current_directory = os.getcwd()

	# Navigate to the parent directory
	parent_directory = os.path.dirname(current_directory)

	# Navigate to the parent of the parent directory (grandparent directory)
	grandparent_directory = os.path.dirname(parent_directory)

	# Path to the directory containing CSV files
	csv_directory = grandparent_directory+'/Datasets/Markov/Exp2/'

	K = len(N_values)

	for i in range(K):
		N = N_values[i]

		output_file_name = './Results/output_'+dist+str(N)+'_fig2.txt'
		futures = []

		with concurrent.futures.ThreadPoolExecutor() as executor:
			for N in N_values:

				# Construct the CSV file name
				csv_file_name = f'markov_ts_{N}_50000_{dist}z.csv'
				csv_file_path = os.path.join(csv_directory, csv_file_name)

				tr_csv_file_name = f'markov_tr_{N}_50000_{dist}z.csv'
				tr_csv_file_path = os.path.join(csv_directory, tr_csv_file_name)
				# Check if the CSV file exists
				if os.path.exists(csv_file_path):
					print(f'Processing {csv_file_name}...')
					# Read values from CSV file
					values = read_csv_values(csv_file_path,n)
					tr_p = read_tr_pvalues(tr_csv_file_path)
				res = executor.submit(process_file, values[:n],tr_p,N)
				futures.append(res)

			# Wait for all futures to complete
			concurrent.futures.wait(futures)

			# Retrieve the results
			results = [future.result() for future in futures]	

		with open(output_file_name,'a') as of:
			for res in results:
				N,HLZ1,HLZ2,Hexact = res
				of.write(f'{N},{HLZ1},{HLZ2},{Hexact}\n')
    	
if __name__ == "__main__":
	main()

