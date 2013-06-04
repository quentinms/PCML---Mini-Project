from data import Data
from mlp import MLP
from error import Error
import matplotlib.pyplot as plt
import scipy as sp
#from pprint import pprint 


class TrainerValidator:

	def __init__(self, k, nb_epochs, H1, H2, nu, mu, batchsize, train_set_size, validation_set_size):
		self.k = k

		self.data = Data(self.k, train_set_size, validation_set_size)
		self.data.importDataFromMat()
		self.data.normalize()
		

		self.mlp = MLP(H1,H2,576, nu, mu, batchsize, self.k, self.data)
		self.error = Error()
		self.NUM_EPOCH = nb_epochs

		self.validation_error = sp.zeros(self.NUM_EPOCH+1)
		self.misclassified_val = sp.zeros(self.NUM_EPOCH+1)
		self.training_error = sp.zeros(self.NUM_EPOCH+1)
		self.misclassified_train = sp.zeros(self.NUM_EPOCH+1)

	def trainAndClassify(self):
		converge = 0
		a = 4
		var_thresh = 0.005
		for i in range(self.NUM_EPOCH+1):
			self.data.shuffleData()
			self.mlp.train()
			_, _, _, _, _, results_train, _, _, _, _, _, _ = self.mlp.forward_pass(self.mlp.data.train_left, self.mlp.data.train_right)
			results_val, results_classif = self.mlp.classify()

			self.training_error[i], self.misclassified_train[i] = self.error.norm_total_error(results_train, self.data.train_cat, self.k)
			self.validation_error[i], self.misclassified_val[i] = self.error.norm_total_error(results_val, self.data.val_cat, self.k)

			print "Epoch #"+str(i)+" Ratio of misclassified: "+str(self.misclassified_val[i])+" - Error: "+str(self.validation_error[i])

			# Early stopping
			if i > 0 :
				if (self.validation_error[i]>(self.validation_error[i-1]*(1-var_thresh))) :
					converge += 1
				else :
					if converge > 0 :
						converge -= 1/2

			if converge>=a :
				print "Triggering early stopping (Cause : increasing or convergence of the error has been detected)"
				break
						
		#self.mlp.test_gradient()

	def plotResults(self):
		error_fig = plt.figure()
		ax1 = error_fig.add_subplot(111)
		ax1.plot(self.validation_error, label='validation error')
		ax1.plot(self.training_error, label='training error')
		#ax1.set_xlim([1,self.NUM_EPOCH])
		ax1.set_ylabel('error')
		ax1.set_xlabel('epoch')
		plt.legend()

		mis_fig = plt.figure()
		ax2 = mis_fig.add_subplot(111)
		ax2.plot(self.misclassified_val, label='misclassified ratio (validation)')
		ax2.plot(self.misclassified_train, label='misclassified ratio (training)')
		#ax2.set_xlim([1,self.NUM_EPOCH])
		ax2.set_ylabel('misclassified')
		ax2.set_xlabel('epoch')
		plt.legend()
		plt.show()
		

		