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
			t = Thread(target=self.createMLPs, args=((i+1)*10, nu, mu, batchsize, k))
			t.start()

	def createMLPs(self, H1, nu, mu, batchsize, k):
		
		for j in range(3) :
			data = Data(k, 0, 0)
			data.importDataFromMat()
			data.normalize()
			train = TrainerValidator(k, 5, H1, (j+1)*10, nu, mu, batchsize, data)
			train.trainAndClassify()
			train.plotResults()
