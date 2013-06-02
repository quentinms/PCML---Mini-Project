from data import Data
from mlp import MLP
from error import Error
import matplotlib.pyplot as plt
import scipy as sp
#from pprint import pprint 

def main():
	data = Data('miniproject_data/norb_binary.mat')
	data.importDataFromMat()
	data.normalize()
	data.shuffleData()
	mlp = MLP(20,50,576, 0.001, 0.1, data)
	error = Error()
	# a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, z2 = mlp.forward_pass(sp.array([1,0]), sp.array([1,0]))
	# mlp.backward_pass(a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, sp.array([1,0]), sp.array([1,0]), z2, 1);
	#mlp.descend(sp.array([[1,0,1], [2,1, 4]]), sp.array([[1,5,0], [2,5, 0]]), sp.array([1,1,1]))

	#mlp.descend(data.train_left[:,1:10], data.train_right[:,1:10],sp.array(data.train_cat[:,1:10], dtype='int8')-2)
	

	NUM_EPOCH = 15

	validation_error = sp.zeros(NUM_EPOCH)
	misclassified_val = sp.zeros(NUM_EPOCH)
	misclassified_train = sp.zeros(NUM_EPOCH)
	training_error = sp.zeros(NUM_EPOCH)

	for i in range(NUM_EPOCH):
		#print "-"*30 + " Training #" +str(i)+ "-"*30
		mlp.train()
		a1L, a1R, a2L, a2LR, a2R, results_train, z1Lb, z1LRb, z1Rb, z2b, xLb, xRb = mlp.forward_pass(data.train_left, data.train_right)
		#print "-"*30 + " Classifying #" +str(i)+ "-"*30
		results_val, results_classif = mlp.classify()

		#print "-"*30 + " Error " + "-"*30
		validation_error[i], misclassified_val[i] = error.total_error(results_val, data.val_cat, 2)
		training_error[i], misclassified_train[i] = error.total_error(results_train, data.train_cat, 2)
		
		print "Epoch #"+str(i)+" Number of misclassified: "+str(misclassified_val[i])+" - Logistic error: "+str(validation_error[i])
		#training_error[i] = error.total_error(results_train, data.train_cat)

	plt.plot(validation_error, label='validation error')
	#plt.plot(misclassified_val, label='misclassified')
	plt.plot(training_error, label='training error')
	plt.ylabel('error')
	plt.xlabel('epoch')
	plt.legend()
	plt.show()
	mlp.test_gradient();	
	#pprint(sp.array(data.val_cat, dtype='int8')-2)

main()

