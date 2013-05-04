from data import Data
from mlp import MLP

def main():
	data = Data('miniproject_data/norb_binary.mat')
	data.importDataFromMat()
	data.normalizeDataset()
	mlp = MLP(4,5,576)
	a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, z2 = mlp.forward_pass(data.train_left, data.train_right)
	print mlp.backward_pass(a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, data.train_left, data.train_right, z2, 1);
main()

