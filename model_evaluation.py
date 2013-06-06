from data import Data
from mlp import MLP
from error import Error
import matplotlib.pyplot as plt
import scipy as sp
from scipy import linalg as LA
import matplotlib.cm as cm
from threading import Thread
from train_validate import TrainerValidator

class ModelEvaluation :

	def __init__(self):
		pass

	def findH1H2(self, nu, mu, batchsize, k):

		for i in range(10) :
			t = Thread(target=self.createMLPsH, args=((i+1)*10, nu, mu, batchsize, k))
			t.start()
			
	def findNuMu(self, H1, H2, batchsize, k):

		nu = [0.001, 0.01, 0.06, 0.1, 0.2, 0.3]
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

	def createMLPsP(self, H1, H2, nu, batchsize, k):
		
		for j in range(10) :
			data = Data(k, 0, 0)
			data.importDataFromMat()
			data.normalize()
			train = TrainerValidator(k, 5, H1, H2, nu, j/10.0, batchsize, data)
			train.trainAndClassify()
			train.plotResults()
