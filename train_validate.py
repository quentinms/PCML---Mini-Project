from data import Data
from mlp import MLP
from error import Error
import matplotlib.pyplot as plt
import scipy as sp
#from pprint import pprint 


class TrainerValidator:

	def __init__(self, k, nb_epochs, H1, H2, nu, mu, test_set_size, validation_set_size):
		self.k = k

		self.data = Data(self.k)
		self.data.importDataFromMat()
		self.data.normalize()
		self.data.shuffleData()

		self.mlp = MLP(H1,H2,576, nu, mu, self.k, self.data)
		self.error = Error()
		self.NUM_EPOCH = nb_epochs

		self.validation_error = sp.zeros(self.NUM_EPOCH+1)
		self.misclassified_val = sp.zeros(self.NUM_EPOCH+1)
		self.training_error = sp.zeros(self.NUM_EPOCH+1)
		self.misclassified_train = sp.zeros(self.NUM_EPOCH+1)

	def trainAndClassify(self):
		"""
		for i in range(self.NUM_EPOCH+1):
			results_train = self.mlp.train()
			results_val, results_classif = self.mlp.classify()
		
			if i != self.NUM_EPOCH :
				self.validation_error[i], self.misclassified_val[i] = self.error.norm_total_error(results_val, self.data.val_cat, self.k)
				print "Epoch #"+str(i+1)+" Ratio of misclassified: "+str(self.misclassified_val[i])+" - Error: "+str(self.validation_error[i])
			if i != 0 :
				self.training_error[i-1], self.misclassified_train[i-1] = self.error.norm_total_error(results_train, self.data.train_cat, self.k)
		"""
		results_val, results_classif = self.mlp.classify()
		for i in range(self.NUM_EPOCH+1):
			
			self.validation_error[i], self.misclassified_val[i] = self.error.norm_total_error(results_val, self.data.val_cat, self.k)
			print "Epoch #"+str(i)+" Ratio of misclassified: "+str(self.misclassified_val[i])+" - Error: "+str(self.validation_error[i])
			
			results_train = self.mlp.train()
			results_val, results_classif = self.mlp.classify()
			
			self.training_error[i], self.misclassified_train[i] = self.error.norm_total_error(results_train, self.data.train_cat, self.k)
		#self.mlp.test_gradient()

	def plotResults(self):
		error_fig = plt.figure()
		ax1 = error_fig.add_subplot(111)
		ax1.plot(self.validation_error, label='validation error')
		ax1.plot(self.training_error, label='training error')
		ax1.set_ylabel('error')
		ax1.set_xlabel('epoch')
		plt.legend()

		mis_fig = plt.figure()
		ax2 = mis_fig.add_subplot(111)
		ax2.plot(self.misclassified_val, label='misclassified ratio (validation)')
		ax2.plot(self.misclassified_train, label='misclassified ratio (training)')
		ax2.set_ylabel('misclassified')
		ax2.set_xlabel('epoch')
		plt.legend()
		plt.show()
		

		