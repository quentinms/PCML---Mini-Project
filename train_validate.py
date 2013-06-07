from data import Data
from mlp import MLP
from error import Error
import matplotlib.pyplot as plt
import scipy as sp
#from pprint import pprint 


class TrainerValidator:

	def __init__(self, k, nb_epochs, H1, H2, nu, mu, batchsize, data):
		self.k = k

		self.data = data

		self.H1 = H1
		self.H2 = H2
		self.mu = mu
		self.nu = nu 
		self.batchsize = batchsize
	
		self.mlp = MLP(H1,H2,576, nu, mu, batchsize, self.k)
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
		early_stopping = 0
		for i in range(self.NUM_EPOCH+1):
			self.data.shuffleData()
			self.mlp.train(self.data.train_left, self.data.train_right, self.data.train_cat)
			_, _, _, _, _, results_train, _, _, _, _, _, _ = self.mlp.forward_pass(self.data.train_left, self.data.train_right)
			results_val, results_classif = self.mlp.classify(self.data.val_left, self.data.val_right)

			self.training_error[i], self.misclassified_train[i] = self.error.norm_total_error(results_train, self.data.train_cat, self.k)
			self.validation_error[i], self.misclassified_val[i] = self.error.norm_total_error(results_val, self.data.val_cat, self.k)

			print "Epoch #"+str(i)+" Ratio of misclassified: "+str(self.misclassified_val[i])+" - Error: "+str(self.validation_error[i])

			
			# Early stopping
			if early_stopping :
				if i > 0 :
					if (self.validation_error[i]>(self.validation_error[i-1]*(1-var_thresh))) :
						converge += 1
					else :
						if converge > 0 :
							converge -= 1/2

				if converge>=a :
					print "Triggering early stopping - Cause : increasing(overfitting) or convergence of the error has been detected"
					break
		#self.mlp.test_gradient(self.data.val_left, self.data.val_right, self.data.val_cat)

	def plotResults(self):
		error_fig = plt.figure()
		ax1 = error_fig.add_subplot(111)
		ax1.plot(self.validation_error, label='validation error')
		ax1.plot(self.training_error, label='training error')
		ax1.set_ylabel('error')
		ax1.set_xlabel('epoch')

		title = "k=%d H1=%d H2=%d mu=%f nu=%f batchsize=%d std(val)=%f std(err)=%f" % (self.k, self.H1, self.H2, self.mu, self.nu, self.batchsize, sp.std(self.validation_error), sp.std(self.training_error) )
		error_fig.suptitle(title)

		plt.legend()

		filename = "k=%d-H1=%d-H2=%d-mu=%f-nu=%f-batchsize=%d-nb_epoch=%d" % (self.k,self.H1, self.H2, self.mu, self.nu, self.batchsize, self.NUM_EPOCH)

		plt.savefig('results/'+filename+"-error.png")
		

		mis_fig = plt.figure()
		ax2 = mis_fig.add_subplot(111)
		ax2.plot(self.misclassified_val, label='misclassified ratio (validation)')
		ax2.plot(self.misclassified_train, label='misclassified ratio (training)')
		title = "k=%d H1=%d H2=%d mu=%f nu=%f batchsize=%d std(val)=%f std(err)=%f" % (self.k, self.H1, self.H2, self.mu, self.nu, self.batchsize, sp.std(self.misclassified_val), sp.std(self.misclassified_train) )
		mis_fig.suptitle(title)
		#ax2.set_xlim([1,self.NUM_EPOCH])
		ax2.set_ylabel('misclassified')
		ax2.set_xlabel('epoch')
		plt.legend()
		plt.savefig('results/'+filename+"-misclassified.png")
		#plt.show()

	def getMLP(self) :
		return self.mlp
		

		