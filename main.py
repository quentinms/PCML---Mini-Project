from train_validate import TrainerValidator


def main():

	test = TrainerValidator(5, 10, 2, 3, 0.001, 0.1, 20, 0,0)
	test.trainAndClassify()
	test.plotResults()

main()

