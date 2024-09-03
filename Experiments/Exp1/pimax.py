import pandas as pd
import sys
sys.path.append('../..') 

from Utils.LZ2 import * 
from Utils.compute_PIMax import * 

import numpy as np
from scipy.optimize import fsolve




def get_data(path, length):
    df = pd.read_csv(path)
    df = df.iloc[:length, :]
    return df.iloc[:, 0].values 


if __name__ == "__main__":

    N_values = [50,45,40,35,30,25,20,15,10,5]
    length = 20000
    n=50000
    a_values = [15,1]
    for a in a_values:
        PM = []
        for N in N_values:
        	
            input_path = "../../Datasets/Markov/Exp1/markov_ts_"+str(N)+"_"+str(n)+"_"+str(a)+".csv"
            series = get_data(input_path, length)
            print("Running... N=",N,"a=",a)
            
            H = Compute_LZ2(series[:int(0.8*len(series))],0.0)
            pimax = get_pimax(H,N)
            print(pimax)
            PM.append(pimax)
            
        
        output_path = "Results/PIMAX/pimax_"+str(N)+"_"+str(n)+"_"+str(a)+".csv"

        with open(output_path,"a") as f:
            for j in range(len(N_values)):
                f.write(str(N)+","+str(N_values[j])+","+str(PM[j])+"\n")
