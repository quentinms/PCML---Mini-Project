from train_validate import TrainerValidator
from data import Data
from model_evaluation import ModelEvaluation
from test import Test
from SquaredErrorLinearClassifier import SquaredErrorLinearClassifier
from LogisticLinearClassifier import LogisticLinearClassifier
import matplotlib.pyplot as plt
import scipy as sp
from error import Error



def main():
	#Binary MLP
	#testBinary()
	#Multi-Way MLP
	#testMultiWay()

	#Squared error linear classifier
	#testSquaredError()

	#Logistic error linear classifier
	testLogisticError()

	#findMuNuLinearClassifier() 

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

def testSquaredError() :
	k=5

	data = Data(k, 0, 0)
	data.importDataFromMat()
	data.normalize()

	sq = SquaredErrorLinearClassifier(2**10, k)
	sq.train(data.train_left, data.train_right, data.train_cat)
	results, cat = sq.classify(data.test_left, data.test_right)
	sq.confusion_matrix(cat, data.test_cat.argmax(axis=0))

	err = Error()
	err, misclass = err.norm_total_error(results.T, data.test_cat, k)
	print "Error on the test set "+str(err)
	print "Misclassification ratio on the test set "+str(misclass)

def testLogisticError() :
	k=5

	data = Data(k, 0, 0)
	data.importDataFromMat()
	data.normalize()

	lg = LogisticLinearClassifier(0.03, 0.03, 576, k, data)
	err_train, miss_train, err_val, miss_val = lg.train(30)
	mis_fig = plt.figure()
	ax2 = mis_fig.add_subplot(111)
	ax2.plot(err_val, label='error (validation)')
	ax2.plot(err_train, label='error (training)')
	title = "std(val)=%f std(err)=%f" % (sp.std(err_val), sp.std(err_train) )
	mis_fig.suptitle(title)
	ax2.set_ylabel('error')
	ax2.set_xlabel('epoch')
	plt.legend()

	mis_fig = plt.figure()
	ax2 = mis_fig.add_subplot(111)
	ax2.plot(miss_val, label='misclassification ratio (validation)')
	ax2.plot(miss_train, label='misclassification ratio (training)')
	mis_fig.suptitle(title)
	ax2.set_ylabel('misclassification ratio')
	ax2.set_xlabel('epoch')
	plt.legend()

	results, cat = lg.classify(data.test_left, data.test_right)
	lg.confusion_matrix(cat, data.test_cat.argmax(axis=0))

	err = Error()
	err, misclass = err.norm_total_error(results.T, data.test_cat, k)
	print "Error on the test set "+str(err)
	print "Misclassification ratio on the test set "+str(misclass)

def findMuNu() :

	k=5

	evalModel = ModelEvaluation()
	evalModel.findNuMu(80, 60, 1, k)

def findMuNuLinearClassifier() :

	k=5

	evalModel = ModelEvaluation()
	evalModel.findNuMuLinearClass(1, k)

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

