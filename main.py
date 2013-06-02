from data import Data
from mlp import MLP
from error import Error
import matplotlib.pyplot as plt
import scipy as sp
#from pprint import pprint 

def main():
	data = Data('miniproject_data/norb_binary.mat', 2)
	data.importDataFromMat()
	data.normalize()
	data.shuffleData()
	mlp = MLP(20,50,576, 0.001, 0.1, 2, data)
	error = Error()
	
	
	NUM_EPOCH = 10

	validation_error = sp.zeros(NUM_EPOCH)
	misclassified_val = sp.zeros(NUM_EPOCH)
	misclassified_train = sp.zeros(NUM_EPOCH)
	training_error = sp.zeros(NUM_EPOCH)
	training_error2 = sp.zeros(NUM_EPOCH)

	for i in range(NUM_EPOCH):
		#print "-"*30 + " Training #" +str(i)+ "-"*30
		results_train = mlp.train()
		_, _, _, _, _, results_train2, _, _, _, _, _, _ = mlp.forward_pass(data.train_left, data.train_right)
		#print "-"*30 + " Classifying #" +str(i)+ "-"*30
		results_val, results_classif = mlp.classify()

		#print "-"*30 + " Error " + "-"*30
		validation_error[i], misclassified_val[i] = error.total_error(results_val, data.val_cat, 2)
		training_error[i], misclassified_train[i] = error.total_error(results_train, data.train_cat, 2)
		
		print "Epoch #"+str(i)+" Number of misclassified: "+str(misclassified_val[i])+" - Logistic error: "+str(validation_error[i])
		training_error2[i], _ = error.total_error(results_train2, data.train_cat, 2)

	plt.plot(validation_error, label='validation error')
	#plt.plot(misclassified_val, label='misclassified')
	plt.plot(training_error, label='training error')
	plt.plot(training_error2, label='new training error')
	plt.ylabel('error')
	plt.xlabel('epoch')
	plt.legend()
	plt.show()
	#mlp.test_gradient();	
	#pprint(sp.array(data.val_cat, dtype='int8')-2)

main()

