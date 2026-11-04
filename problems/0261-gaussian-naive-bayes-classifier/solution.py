import numpy as np
import math

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
	"""
	Implements Gaussian Naive Bayes classifier.
	
	Args:
		X_train: Training features (shape: N_train x D)
		y_train: Training labels (shape: N_train)
		X_test: Test features (shape: N_test x D)
	
	Returns:
		Predicted class labels for X_test (shape: N_test)
	"""
	# Your code here

	classes = np.unique(y_train)
	scores = []

	for i in classes:
		X_class = X_train[y_train == i]

		mean = np.mean(X_class, axis = 0)
		var = np.var(X_class, axis=0)

		prior = len(X_class)/len(y_train)
		log_prior = np.log(prior)

		log_likelihood = -0.5*np.sum(np.log(2*np.pi*var) + ((X_test-mean)**2)/var, axis=1)

		scores.append(log_likelihood + log_prior)
	
	scores = np.array(scores).T
	preds = np.argmax(scores, axis=1)

	return preds