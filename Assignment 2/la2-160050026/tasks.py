import nn
import numpy as np
import sys

from util import *
from visualize import *
from layers import *


# XTrain - List of training input Data
# YTrain - Corresponding list of training data labels
# XVal - List of validation input Data
# YVal - Corresponding list of validation data labels
# XTest - List of testing input Data
# YTest - Corresponding list of testing data labels

def taskSquare(draw):
	XTrain, YTrain, XVal, YVal, XTest, YTest = readSquare()
	# print(XTrain.shape, YTrain.shape, XVal.shape, YVal.shape, XTest.shape, YTest.shape)
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.1 - YOUR CODE HERE
	nn1 = nn.NeuralNetwork(2, 0.1, 50, 40)
	nn1.addLayer(FullyConnectedLayer(2,8))
	nn1.addLayer(FullyConnectedLayer(8,2))
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	# Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 2'
	# Use drawSquare(XTest, pred) to visualize YOUR predictions.
	if draw:
		drawSquare(XTest, pred)
	return nn1, XTest, YTest


def taskSemiCircle(draw):
	XTrain, YTrain, XVal, YVal, XTest, YTest = readSemiCircle()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.2 - YOUR CODE HERE
	nn1 = nn.NeuralNetwork(2, 0.1, 20, 10)
	nn1.addLayer(FullyConnectedLayer(2, 3))
	nn1.addLayer(FullyConnectedLayer(3, 3))
	nn1.addLayer(FullyConnectedLayer(3, 2))
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc  = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	# Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 4'
	# Use drawSemiCircle(XTest, pred) to vnisualize YOUR predictions.
	if draw:
		drawSemiCircle(XTest, pred)
	return nn1, XTest, YTest

def taskMnist():
	XTrain, YTrain, XVal, YVal, XTest, YTest = readMNIST()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.3 - YOUR CODE HERE
	nn1 = nn.NeuralNetwork(10, 0.1, 20, 40)
	nn1.addLayer(FullyConnectedLayer(784, 50))
	nn1.addLayer(FullyConnectedLayer(50, 12))
	nn1.addLayer(FullyConnectedLayer(12, 10))	
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc  = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	return nn1, XTest, YTest

def taskCifar10():	
	XTrain, YTrain, XVal, YVal, XTest, YTest = readCIFAR10()
	
	XTrain = XTrain[0:5000,:,:,:]
	XVal = XVal[0:1000,:,:,:]
	XTest = XTest[0:1000,:,:,:]
	YVal = YVal[0:1000,:]
	YTest = YTest[0:1000,:]
	YTrain = YTrain[0:5000,:]
	
	modelName = 'model.npy'
	# # Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# # nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# # Add layers to neural network corresponding to inputs and outputs of given data
	# # Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	# ###############################################
	# # TASK 2.4 - YOUR CODE HERE
	nn1 = nn.NeuralNetwork(10, 0.1, 20, 30)
	# nn1.addLayer(ConvolutionLayer([3,32,32], [6,6], 4, 2))
	# nn1.addLayer(AvgPoolingLayer([4, 14, 14], [2,2], 2))
	# nn1.addLayer(ConvolutionLayer([4,7,7], [3,3], 2, 2))
	# nn1.addLayer(FlattenLayer())
	# nn1.addLayer(FullyConnectedLayer(18, 10))
	# 35.0

	#model1
	# nn1.addLayer(ConvolutionLayer([3,32,32], [16,16], 10, 8))
	# nn1.addLayer(FlattenLayer())
	# nn1.addLayer(FullyConnectedLayer(90, 20))
	# nn1.addLayer(FullyConnectedLayer(20, 10))
	#37.6 seed=735 epoch=30

	#model2
	nn1.addLayer(ConvolutionLayer([3,32,32], [16,16], 10, 8))
	nn1.addLayer(FlattenLayer())
	nn1.addLayer(FullyConnectedLayer(90, 10))
	#37.9 seed=735 epoch=30

	#model3
	# nn1.addLayer(ConvolutionLayer([3,32,32], [16,16], 5, 8))
	# nn1.addLayer(FlattenLayer())
	# nn1.addLayer(FullyConnectedLayer(45, 10))
	# nn1.addLayer(FullyConnectedLayer(20, 10))
	#34.5 seed=231 epoch=60

	#model4
	# nn1.addLayer(ConvolutionLayer([3,32,32], [16,16], 6, 8))
	# nn1.addLayer(FlattenLayer())
	# nn1.addLayer(FullyConnectedLayer(54, 10))
	#36% seed=735 epoch=40

	###################################################
	return nn1,  XTest, YTest, modelName # UNCOMMENT THIS LINE WHILE SUBMISSION


	nn1.train(XTrain, YTrain, XVal, YVal, True, True, loadModel=False, saveModel=True, modelName=modelName)
	pred, acc = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)