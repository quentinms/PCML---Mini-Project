from train_validate import TrainerValidator
from data import Data
from model_evaluation import ModelEvaluation
from test import Test
import matplotlib.pyplot as plt
import scipy as sp



def main():

	k=5
	"""
	data = Data(k, 0, 0)
	data.importDataFromMat()
	data.normalize()
	"""

	evalModel = ModelEvaluation()
	#evalModel.findH1H2(0.001, 0.1, 1, k)

	v_parameters = sp.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 259])
	evalModel.findV(v_parameters, 2, k)

	"""
	train = TrainerValidator(k, 2, 20, 50, 0.001, 0.1, 1, data)
	train.trainAndClassify()
	train.plotResults()

	test = Test(train.getMLP(), data, k)
	test.classify()
	#test.examples()
	test.plot_confusion_matrix()
	"""
	plt.show()



main()

