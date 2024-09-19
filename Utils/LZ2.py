import numpy as np
def bisect_left(arr, x, lo=0, hi=None):
    """
    Locate the insertion point for x in a sorted sequence arr.
    The parameters lo and hi may be used to specify a subset of the sequence
    to search.
    """
    if hi is None:
        hi = len(arr)

    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid

    return lo

def bisect_right(arr, x, lo=0, hi=None):
    """
    Return the index after the last occurrence of x in a sorted sequence arr.
    The parameters lo and hi may be used to specify a subset of the sequence
    to search.
    """
    if hi is None:
        hi = len(arr)

    while lo < hi:
        mid = (lo + hi) // 2
        if x < arr[mid]:
            hi = mid
        else:
            lo = mid + 1

    return lo
    
def overlap(x0,x1,e):

	st1, ed1 = (x0-e,x0+e)
	st2, ed2 = (x1-e,x1+e)
	
	if ed1 >= st2 and ed2 >= st1:
		return True
	
	return False
    
def LZ2lookup(D, s, j, e):
    # Initialize an empty list to store the result indexes
    matches = []

    # Get the sorted keys from the dictionary
    keys = list(D.keys())

    # Use binary search to find the left and right bounds of keys within the tolerance range
    left_bound = bisect_left(keys, s - 2*e)
    right_bound = bisect_right(keys, s + 2*e)

    # Iterate over the relevant keys within the tolerance range
    for key in keys[left_bound:right_bound]:
        # Get the indexes associated with the current key
        values = D[key]

        # Use binary search to find the index where j should be inserted in the values
        index = bisect_left(values, j)

        # Extend the result indexes with the relevant values
        matches.extend(values[:index])

    # Return the final list of indexes
    return matches

def LZ2build_IIdx(T):
    index = {}

    # Build the dictionary index
    for i, char in enumerate(T):
        index.setdefault(char, []).append(i)

    # Create a sorted dictionary by keys
    sorted_index = dict(sorted(index.items()))

    # Return the inverted index
    return sorted_index
    
def LZ2_adapted(T, e):
    # Get the length of the input sequence T
    n = len(T)
    
    # Initialize an array to store the lambda values
    lambdai = np.zeros(n)
    
    # Build the inverted index for quick lookup
    Idx = LZ2build_IIdx(T)
    
    # Iterate over each position in the sequence
    for cur in range(n):
        lambda_max = 0
        
        # Return the indices i < cur where T[cur] ~e T[i]
        matches = LZ2lookup(Idx, T[cur], cur, e)
        
        # Iterate over matching indices
        for i in matches:
            lambda_temp = 0
            
            # Compare elements within the tolerance until a mismatch is found
            while (cur + lambda_temp < n) and (i + lambda_temp < cur):
                if overlap(T[i + lambda_temp], T[cur + lambda_temp],e):
                    lambda_temp += 1
                else:
                    break
            
            # Update the maximum lambda value
            if lambda_temp > lambda_max:
                lambda_max = lambda_temp
            
            # If the end of the sequence is reached, return the current lambdai
            if cur + lambda_temp >= n:
                return lambdai
        
        # Set the lambda value for the current position
        lambdai[cur] = lambda_max + 1
    
    # Return the final array of lambda values
    return lambdai

def Compute_LZ2(T,e):

	lambdai = LZ2_adapted(T,e)
	
	H = len(T) * np.log2(len(T)) /(sum(lambdai))
	
	return H
    

	
	
		


