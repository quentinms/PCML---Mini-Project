from train_validate import TrainerValidator
from data import Data
from model_evaluation import ModelEvaluation
from test import Test
import matplotlib.pyplot as plt
import scipy as sp



def main():

	testBinary()
	testMultiWay()
	
	plt.show()

def testBinary() :
	k=2

	data = Data(k, 0, 0)
	data.importDataFromMat()
	data.normalize()

	train = TrainerValidator(k, 70, 100, 10, 0.1, 0.2, 1, data)
	train.trainAndClassify()
	train.plotResults()

	test = Test(train.getMLP(), data, k)
	test.classify()
	test.examples()
	test.plot_confusion_matrix()

def testMultiWay() :

	k=5

	data = Data(k, 0, 0)
	data.importDataFromMat()
	data.normalize()

	train = TrainerValidator(k, 70, 80, 60, 0.004, 0.1, 1, data)
	train.trainAndClassify()
	train.plotResults()

	test = Test(train.getMLP(), data, k)
	test.classify()
	test.examples()
	test.plot_confusion_matrix()

def findMuNu() :

	k=5
	evalModel = ModelEvaluation()
	evalModel.findNuMu(80, 60, 1, k)

def findH1H2() :

	k=2
	evalModel = ModelEvaluation()
	evalModel.findNuMu(0.001, 0.1, 1, k)

def compareParameters() :

	k=5

	data = Data(k, 0, 0)
	data.importDataFromMat()
	data.normalize()

	train = TrainerValidator(k, 40, 80, 60, 0.001, 0.1, 1, data)
	train.trainAndClassify()
	train2 = TrainerValidator(k, 40, 80, 60, 0.04, 0.1, 1, data)
	train2.trainAndClassify()
	train3 = TrainerValidator(k, 40, 80, 60, 0.1, 0.1, 1, data)
	train3.trainAndClassify()

	error_fig = plt.figure()
	ax1 = error_fig.add_subplot(111)
	ax1.plot(train.validation_error, label='validation error mu=0.1 nu=0.001')
	ax1.plot(train.training_error, label='training error mu=0.1 nu=0.001')
	ax1.plot(train2.validation_error, label='validation error mu=0.1 nu=0.04')
	ax1.plot(train2.training_error, label='training error mu=0.1 nu=0.04')
	ax1.plot(train3.validation_error, label='validation error mu=0.1 nu=0.1')
	ax1.plot(train3.training_error, label='training error mu=0.1 nu=0.1')
	ax1.set_ylabel('error')
	ax1.set_xlabel('epoch')

	title = "Validation and training errors k=5 H1=80 H2=60 batchsize=1"
	error_fig.suptitle(title)

	plt.legend()

main()

