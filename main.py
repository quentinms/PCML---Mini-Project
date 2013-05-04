from data import Data
from mlp import MLP

def main():
	data = Data('miniproject_data/norb_binary.mat')
	data.importDataFromMat()
	data.normalizeDataset()
	mlp = MLP(600,300,576)
	# mlp.forward_pass(data.train_left, data.train_right)
	mlp.forward_pass(data.train_left, data.train_right)
main()

