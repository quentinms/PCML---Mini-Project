from data import Data
from mlp import MLP
#import scipy as sp
#from pprint import pprint 

def main():
	data = Data('miniproject_data/norb_binary.mat')
	data.importDataFromMat()
	data.normalize()
	mlp = MLP(100,75,576, 0.1, 0.1, data)
	# a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, z2 = mlp.forward_pass(sp.array([1,0]), sp.array([1,0]))
	# mlp.backward_pass(a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, sp.array([1,0]), sp.array([1,0]), z2, 1);
	#mlp.descend(sp.array([[1,0,1], [2,1, 4]]), sp.array([[1,5,0], [2,5, 0]]), sp.array([1,1,1]))

	#mlp.descend(data.train_left[:,1:10], data.train_right[:,1:10],sp.array(data.train_cat[:,1:10], dtype='int8')-2)
	

	print "-"*30 + " Training " + "-"*30
	mlp.train()
	print "-"*30 + " Classifying " + "-"*30
	mlp.classify()
		
	#pprint(sp.array(data.val_cat, dtype='int8')-2)
main()

