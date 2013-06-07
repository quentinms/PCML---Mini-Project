from train_validate import TrainerValidator
from data import Data
from model_evaluation import ModelEvaluation
from test import Test
import matplotlib.pyplot as plt
import scipy as sp



def main():

	k=5
	"""
	data = Data(k, 20, 20)
	data.importDataFromMat()
	data.normalize()
	"""

	evalModel = ModelEvaluation()
	evalModel.findNuMu(80, 60, 1, k)
	"""
	train = TrainerValidator(k, 160, 100, 10, 0.1, 0, 1, data)
	train.trainAndClassify()
	train.plotResults()

	test = Test(train.getMLP(), data, k)
	test.classify()
	#test.examples()
	test.plot_confusion_matrix()
	
	plt.show()
	"""


main()

