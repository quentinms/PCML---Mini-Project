from train_validate import TrainerValidator
from data import Data
from test import Test


def main():

	k=5

	data = Data(k, 0, 0)
	data.importDataFromMat()
	data.normalize()

	train = TrainerValidator(k, 20, 20, 50, 0.001, 0.1, 1, data)
	train.trainAndClassify()
	train.plotResults()

	test = Test(train.getMLP(), data, k)
	test.classify()
	#test.examples()
	test.confusion_matrix()



main()

