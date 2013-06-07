from data import Data
from mlp import MLP
from error import Error
import matplotlib.pyplot as plt
import scipy as sp
from scipy import linalg as LA
import matplotlib.cm as cm
from threading import Thread
from train_validate import TrainerValidator
from SquaredErrorLinearClassifier import SquaredErrorLinearClassifier

class ModelEvaluation :

	def __init__(self):
		pass

	def findH1H2(self, nu, mu, batchsize, k):

		for i in range(10) :
			t = Thread(target=self.createMLPsH, args=((i+1)*10, nu, mu, batchsize, k))
			t.start()
			
	def findNuMu(self, H1, H2, batchsize, k):

		nu = [0.001, 0.01, 0.04, 0.08, 0.1]
		for i in nu :
			t = Thread(target=self.createMLPsP, args=(H1, H2, i, batchsize, k))
			t.start()

	def createMLPsH(self, H1, nu, mu, batchsize, k):
		
		for j in range(10) :
			data = Data(k, 0, 0)
			data.importDataFromMat()
			data.normalize()
			train = TrainerValidator(k, 5, H1, (j+1)*10, nu, mu, batchsize, data)
			train.trainAndClassify()
			train.plotResults()

	def findV(self, v_parameters, fold, k):

		for idx, v in enumerate(v_parameters):
			t = Thread(target=self.crossvalidation, args=(fold, v, k))
			t.start()

	def crossvalidation(self, fold, v, k):
		
		error = 0
		data = Data(k, 0, 0)
		data.importDataFromMat()
		data.normalize()

		n = data.train_cat.shape[1]
		all_indices = sp.arange(n)
		data.shuffleData()

		sq = SquaredErrorLinearClassifier(v,k)

		dataset_indices = sp.split(all_indices, fold)

		for i in range(fold):
			set_without_D_i_indices = sp.concatenate(dataset_indices[0:i]+dataset_indices[i+1:fold])
			
			#print "-"*30+"train"+"-"*30
			sq.train(data.train_left[:,set_without_D_i_indices], data.train_right[:,set_without_D_i_indices], data.train_cat[:,set_without_D_i_indices])
			#print "-"*30+"classify"+"-"*30
			results, classes = sq.classify(data.train_left[:,dataset_indices[i]], data.train_right[:,dataset_indices[i]])
			#print "-"*30+"error"+"-"*30

			err,  _ = sq.error(results, data.train_cat[:,dataset_indices[i]].T)

			error += fold/(n*1.0)*err

		error = error / fold

		with open("results/crossvalidation.txt", "a") as myfile:
			toWrite = "v="+str(v)+" error="+str(error)
			myfile.write(toWrite)

	def createMLPsP(self, H1, H2, nu, batchsize, k):
		
		for j in range(4,8) :
			data = Data(k, 0, 0)
			data.importDataFromMat()
			data.normalize()
			train = TrainerValidator(k, 50, H1, H2, nu, j/10.0, batchsize, data)
			train.trainAndClassify()
			train.plotResults()

