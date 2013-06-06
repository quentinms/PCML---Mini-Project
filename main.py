from train_validate import TrainerValidator
from data import Data
from model_evaluation import ModelEvaluation
from test import Test
import matplotlib.pyplot as plt
import scipy as sp



def main():

	k=2
	"""
	data = Data(k, 0, 0)
	data.importDataFromMat()
	data.normalize()
	"""

	evalModel = ModelEvaluation()
	evalModel.findNuMu(100, 10, 1, k)
	"""
	train = TrainerValidator(k, 160, 30, 60, 0.001, 0.1, 1, data)
	train.trainAndClassify()
	train.plotResults()

	test = Test(train.getMLP(), data, k)
	test.classify()
	#test.examples()
	test.plot_confusion_matrix()
	
	plt.show()
	"""


main()

