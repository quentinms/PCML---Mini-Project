from train_validate import TrainerValidator


def main():

	test = TrainerValidator(5, 10, 20, 50, 0.001, 0.1, 0,0)
	test.trainAndClassify()
	test.plotResults()

main()

