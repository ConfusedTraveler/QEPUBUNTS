import numpy as np

def exact_H(S, N, transition_probs, initial_probs):
	H = 0
	H = sum(-ini * np.log2(ini) for ini in initial_probs)

	for i in range(1, len(S)):
		h = 0
		for j in range(N):
			p = transition_probs[S[i-1], j]
			if p != 0:
				h = h - p * np.log2(p)
		H = H + h

	return H / len(S)


def get_exactH(N, P):
	U = solve_equation(P) #get_U(P) 
	H = 0
	for i in range(N):
		h = 0
		for j in range(N):
			if P[i][j] > 0:
				h -= U[i] * P[i][j] * np.log2(P[i][j])
		H += h
	return H

def solve_equation(P):
	N = len(P)  # Size of the matrix
	U = np.ones(N) / N  # Initial guess for U with equal values

	# Iterative solution using np.linalg.solve
	while True:
		U_new = np.dot(U, P)  # Calculate U_new = U * P
		if np.allclose(U, U_new):  # Check if U and U_new are close
			break
		U = U_new

	# Normalize U to have a sum of 1
	U /= np.sum(U)

	return U
	
def get_U(P):

	# Calculate the stationary distribution
	eigenvalues, eigenvectors = np.linalg.eig(P.T)
	stationary_index = np.argmin(np.abs(eigenvalues - 1.0))
	stationary_distribution = np.abs(eigenvectors[:, stationary_index])
	stationary_distribution /= np.sum(stationary_distribution)
	
	return stationary_distribution
