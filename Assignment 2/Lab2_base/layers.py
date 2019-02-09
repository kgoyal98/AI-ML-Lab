import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE
		self.data = sigmoid(np.matmul(X, self.weights)+self.biases)
		return self.data
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		# print("hi", self.data.shape, delta.shape)
		delta1 = deriv(self.data)*delta
		x = np.matmul(delta1, np.transpose(self.weights))
		self.weights = self.weights - lr*np.matmul(np.transpose(activation_prev), delta1)
		self.biases = self.biases - lr*np.sum(delta1, axis=0)
		return x
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		X1=[]
		for i in range(n):
			x = X[i,:,:,:]
			F=[]
			for f in range(self.out_depth):
				y=[]
				w = self.weights[f,:,:,:]
				for j in range(self.out_row):
					for k in range(self.out_col):
						y.append(sigmoid(np.sum(x[:,j*self.stride:j*self.stride+self.filter_row, k*self.stride:k*self.stride+self.filter_col]*w) + self.biases[f]))
				y = np.asarray(y)
				y = np.reshape(y, [self.out_row,self.out_col])
				F.append(y)
			F = np.asarray(F)
			X1.append(F)
		X1 = np.asarray(X1)
		self.data = X1
		return X1


		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		x = np.zeros([n, self.in_depth, self.in_row, self.in_col])
		dweight = np.zeros(self.weights.shape)
		delta1  = deriv(self.data)*delta
		# print(delta1.shape)
		for i in range(n):
			for f in range(self.out_depth):
				for l in range(self.out_row):	
					for m in range(self.out_col):
						# print(dweight[f,:,:,:].shape, activation_prev[i,:,self.stride*l:self.stride*l+self.filter_row, self.stride*m:self.stride*m+self.filter_col].shape)
						dweight[f,:,:,:] += activation_prev[i,:,self.stride*l:self.stride*l+self.filter_row, self.stride*m:self.stride*m+self.filter_col]*delta1[i,f,l,m]
						self.biases[f]-= lr*delta1[i,f,l,m]
						x[i,:,self.stride*l:self.stride*l+self.filter_row, self.stride*m:self.stride*m+self.filter_col] += self.weights[f,:,:,:]*delta1[i,f,l,m]
		
		self.weights -= lr*dweight
		return x
		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_channels[0] X self.in_channels[1] X self.in_channels[2]]
		# OUTPUT activation matrix		:[n X self.outputsize[0] X self.outputsize[1] X self.in_channels[2]]

		###############################################
		# TASK 1 - YOUR CODE HERE
		X1=[]
		for i in range(n):
			x = X[i,:,:,:]
			D=[]
			for d in range(self.out_depth):
				y=[]
				for j in range(self.out_row):
					for k in range(self.out_col):
						y.append(np.sum(x[d,j*self.stride:j*self.stride+self.filter_row, k*self.stride:k*self.stride+self.filter_col])/(self.filter_row*self.filter_col))
				y = np.asarray(y)
				y = np.reshape(y, [self.out_row,self.out_col])
				D.append(y)
			D = np.asarray(D)
			X1.append(D)
		X1 = np.asarray(X1)
		self.data=X1
		return X1
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		# delta1  = deriv(self.data)*delta
		# print("sdf", delta.shape)
		x = np.zeros([n, self.in_depth, self.in_row, self.in_col])
		for i in range(n):
			for f in range(self.out_depth):
				for l in range(self.out_row):
					for m in range(self.out_col):
						x[i, :,self.stride*l:self.stride*l+self.filter_row, self.stride*m:self.stride*m+self.filter_col] += delta[i,f,l,m]/(self.filter_row*self.filter_col)	
		return x
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))

def deriv(x):
	return x*(1-x)