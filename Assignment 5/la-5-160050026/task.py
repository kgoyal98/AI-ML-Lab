import numpy as np
from utils import *

def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''

	new_X = np.ones((X.shape[0], 1))
	for i in range(1, X.shape[1]):
		column = X[:,[i]]
		if(type(column[0][0])==str):
			new_column = one_hot_encode(column, list(set(column.flatten())))
			new_X = np.append(new_X, new_column, axis=1)
		else:
			data = column.flatten()
			new_column = (column - np.mean(data))/np.std(data)
			new_X = np.append(new_X, new_column, axis=1)
	new_X = new_X.astype(float)
	Y = Y.astype(float)
	return new_X, Y


def grad_ridge(W, X, Y, _lambda):
	'''  TASK 2
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''
	return 2*(np.transpose(X) @ (X @ W - Y) + _lambda*W);

def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-4):
	''' TASK 2
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompute some values to make computation faster
	'''
	(n, d) = X.shape
	W = np.random.normal(0,0.01,(d, 1))
	for i in range(max_iter):
		grad = grad_ridge(W, X, Y, _lambda);
		if(np.linalg.norm(grad, ord=2) < epsilon):
			break
		W -= lr*grad
	return W

def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' TASK 3
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''
	sse_list=[]
	n=X.shape[0]
	for l in lambdas:
		sse_k = 0
		nk = int(n/k)
		for i in range(k):
			X_train = np.append(X[:i*nk, :], X[(i+1)*nk:, :], axis=0)
			Y_train = np.append(Y[:i*nk, :], Y[(i+1)*nk:, :], axis=0)
			X_test = X[i*nk:(i+1)*nk, :]
			Y_test = Y[i*nk:(i+1)*nk, :]
			W = algo(X_train, Y_train, l)
			sse_k += sse(X_test, Y_test, W)
		sse_k/=k
		sse_list.append(sse_k)
	return sse_list


def coord_grad_descent(X, Y, _lambda, max_iter=1000):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	'''
	(n, d) = X.shape
	W = np.random.normal(0,0.2,(d, 1))
	xty = np.transpose(X) @ Y
	xtx = np.transpose(X) @ X
	for _ in range(max_iter):
		for i in range(d):
			y = xty[[i], :]
			x = xtx[[i], :]
			y = x @ W - x[0,i]*W[i,0] - y
			if(x[0,i]!=0):
				if (-(y[0,0]+_lambda/2)/x[0,i])>0:
					W[i,0] = -(y[0,0]+_lambda/2)/x[0,i]
				elif (-(y[0,0]-_lambda/2)/x[0,i])<0:
					W[i,0] = -(y[0,0]-_lambda/2)/x[0,i]
				else:
					W[i,0]=0
			else:
				W[i,0] = 0
	return W


if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	
	lambdas = [300000, 325000, 350000, 375000, 400000] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, coord_grad_descent)
	plot_kfold(lambdas, scores)