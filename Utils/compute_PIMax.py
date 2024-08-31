import numpy as np
from scipy.optimize import fsolve

def get_pimax(S, N):
    func = lambda x: -x * np.log2(x) - (1 - x) * np.log2(1 - x) + (1 - x) * (np.log2(N - 1)) - S
    result = fsolve(func, 0.99999)
    return result[0]
    
def get_pimax_numeric(S, N):
    func = lambda x: -x * np.log2(x) - (1 - x) * np.log2(1 - x) + (1 - x) * (np.log2(N - 2)) - S
    result = fsolve(func, 0.99999)
    return result[0]

    
def get_cn(S, n):
    func = lambda x: x + x*np.log2(x) - n*S
    result = fsolve(func, 100)
    return result[0]/n
    
def get_sum_lambda(S, n):
    sum_lambda = S/np.log2(n)
    return sum_lambda
