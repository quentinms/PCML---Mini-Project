import numpy as np
import scipy as sp
import scipy.io as spio
from matplotlib import pyplot as plt

def importDataFromMat(fileName):
	tmp = spio.loadmat(fileName)
	
	size=tmp['train_cat_s'].shape[1]
	training_set=np.random.choice(size, (2*size/3), False)
	complete_set=np.arange(size)
	val_set=np.setdiff1d(complete_set,training_set);
	
	#Training Data
	train_cat_s=tmp['train_cat_s'][:,training_set]
	train_left_s=tmp['train_left_s'][:,training_set]
	train_right_s=tmp['train_right_s'][:,training_set]

	#Validation Data
	val_cat_s=tmp['train_cat_s'][:,val_set]
	val_left_s=tmp['train_left_s'][:,val_set]
	val_right_s=tmp['train_right_s'][:,val_set]

	#Test Data
	test_cat_s=tmp['test_left_s']
	test_left_s=tmp['test_left_s']
	test_right_s=tmp['test_right_s']
	
	
	#Show image
	#plt.imshow(sp.reshape(test_left_s[:,30],(24,24)))
	#plt.show()


importDataFromMat('miniproject_data/norb_binary.mat')
