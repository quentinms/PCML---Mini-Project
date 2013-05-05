from data import Data
from mlp import MLP
import scipy as sp

def main():
	data = Data('miniproject_data/norb_binary.mat')
	data.importDataFromMat()
	data.normalizeDataset()
	mlp = MLP(4,5,1)
	a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, z2 = mlp.forward_pass(sp.array([1,0]), sp.array([1,0]))
	mlp.backward_pass(a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, sp.array([1,0]), sp.array([1,0]), z2, 1);
main()

