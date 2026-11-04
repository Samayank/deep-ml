import numpy as np

def calculate_correlation_matrix(X, Y=None):
	# Your code here
	import math

	if Y is None:
		Y = X
	sdX = np.sqrt(np.var(X,axis=0,ddof=1))
	sdY = np.sqrt(np.var(Y,axis=0,ddof=1))
	cov = np.cov(X.T,Y.T)

	n = X.shape[1]
	m = Y.shape[1]

	cov = cov[:n, n:n+m]
	corr = cov/(sdX[:,None]*sdY[None,:])
	return corr