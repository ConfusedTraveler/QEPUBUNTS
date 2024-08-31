import pandas as pd
import sys
sys.path.append('../../') 

from Utils.LZ2 import * 
from Utils.compute_PIMax import *

import numpy as np
from scipy.optimize import fsolve
import os
from markov_predict import *
from AR1 import *



def get_data(path, length):
    df = pd.read_csv(path)
    df = df.iloc[:length, :]
    return df.iloc[:length, 0].values 


if __name__ == "__main__":

    # Get the current directory of the script
    current_directory = os.getcwd()
    
    # Navigate to the parent directory
    parent_directory = os.path.dirname(current_directory)
    
    # Navigate to the parent of the parent directory (grandparent directory)
    grandparent_directory = os.path.dirname(parent_directory)
    

    N_values = [5,10,15,20,25,30,35,40,45,50]
    n = 10000
    a_values = [15,1]
    for a in a_values:
        PM = []
        for N in N_values:
            # Path to the CSV file in the grandparent directory
            input_path = grandparent_directory + "/Datasets/Markov/Exp1/markov_ts_"+str(N)+"_"+str(50000)+"_"+str(a)+".csv"
            
            series = get_data(input_path, n)
            output_path_markov = "Results/Markov/markov_pred_"+str(N)+"_"+str(n)+"_"+str(a)+".csv"
            predict(series,N,output_path_markov)
            output_path_ar1 = "Results/AR1/ar1_pred_"+str(N)+"_"+str(n)+"_"+str(a)+".csv"
            model_train_pred(series,output_path_ar1,N)
            H = Compute_LZ2(series[:int(0.8*len(series))],0.4)
            pimax = get_pimax(H,N)
            print(pimax)
            PM.append(pimax)
        
        output_path = "Results/PIMAX/pimax"+"_"+str(n)+"_"+str(a)+".csv"

        with open(output_path,"a") as f:
            for j in range(len(N_values)):
                f.write(str(n)+","+str(N_values[j])+","+str(PM[j])+"\n")
