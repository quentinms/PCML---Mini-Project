from train_validate import TrainerValidator


def main():

	test = TrainerValidator(2, 300, 20, 50, 0.01, 0, 10, 50, 100)
	test.trainAndClassify()
	test.plotResults()

main()

