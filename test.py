import numpy as np
import scipy as sp
import scipy.io as spio
from matplotlib import pyplot as plt

def importDataFromMat(fileName):
	tmp = spio.loadmat(fileName)

	size=tmp['train_cat_s'].shape[1]
	
	## TODO Randomize split

	#Training Data
	train_cat_s=tmp['train_cat_s'][:,:(2*size/3)]
	train_left_s=tmp['train_left_s'][:,:(2*size/3)]
	train_right_s=tmp['train_right_s'][:,:(2*size/3)]

	#Validation Data
	val_cat_s=tmp['train_cat_s'][:,(2*size/3):]
	val_left_s=tmp['train_left_s'][:,(2*size/3):]
	val_right_s=tmp['train_right_s'][:,(2*size/3):]

	#Test Data
	test_cat_s=tmp['test_left_s']
	test_left_s=tmp['test_left_s']
	test_right_s=tmp['test_right_s']

	print train_right_s.shape
	print val_right_s.shape
	
	
	#Show image
	#plt.imshow(sp.reshape(test_left_s[:,30],(24,24)))
	#plt.show()


importDataFromMat('miniproject_data/norb_binary.mat')