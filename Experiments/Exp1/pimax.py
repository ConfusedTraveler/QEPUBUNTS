import pandas as pd
import sys
sys.path.append('../..') 

from Utils.LZ2 import * 

import numpy as np
from scipy.optimize import fsolve




def get_data(path, length):
    df = pd.read_csv(path)
    df = df.iloc[:length, :]
    return df.iloc[:, 0].values 


if __name__ == "__main__":

    N_values = [5,10,15,20,25,30,35,40,45,50]
    length = 10000
    n=50000
    a_values = [1,15]
    for a in a_values:
        PM = []
        for N in N_values:
            input_path = "../../Datasets/Markov/Exp1/markov_ts_"+str(N)+"_"+str(n)+"_"+str(a)+".csv"
            series = get_data(input_path, length)
            H = Compute_LZ2(series[:int(0.8*len(series))],0.0)
            pimax = get_pimax(H,N)
            print(pimax)
            PM.append(pimax)
        
        output_path = "Results/PIMAX/pimax_"+str(N)+"_"+str(n)+"_"+str(a)+".csv"

        with open(output_path,"a") as f:
            for j in range(len(N_values)):
                f.write(str(N)+","+str(N_values[j])+","+str(PM[j])+"\n")
