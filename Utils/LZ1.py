import numpy as np
def get_distinct_words(sequence, epsilon):
		
    dictionary = {}
    word = []
    
    for number in sequence:
        word.append(number)
        
        if len(word) not in dictionary:
            dictionary[len(word)] = [word.copy()]
            word.clear()
        elif not check_presence(word, dictionary[len(word)], epsilon):
            dictionary[len(word)].append(word.copy())
            word.clear()
    
    cn = sum(len(words_list) for words_list in dictionary.values())
    return cn

def check_presence(Q, LW, e):
    for sublist in LW:
        if check_epsilon_overlap(Q, sublist, e):
            return True
    return False

def check_epsilon_overlap(L1, L2, epsilon):
    for i in range(len(L1)):
        interval1 = (L1[i] - epsilon, L1[i] + epsilon)
        interval2 = (L2[i] - epsilon, L2[i] + epsilon)
        if interval1[1] < interval2[0] or interval1[0] > interval2[1]: #abs(L1[i]-L2[i]) > epsilon: #
            return False
	
    return True

def Compute_LZ1(data, epsilon):
    cn = get_distinct_words(data, epsilon) #number of distinct words in dictionary
    n = len(data)
    
    entropy_rate = ((cn) * (np.log2(cn) + 1)) / n
    return entropy_rate

