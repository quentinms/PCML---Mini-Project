import numpy as np
import scipy as sp
import scipy.io as spio
from matplotlib import pyplot as plt

class Data:

	def __init__(self, fileName):
		self.fileName = fileName

	def importDataFromMat(self):
		tmp = spio.loadmat(self.fileName)

		print "Importing", self.fileName+"...",

		size=tmp['train_cat_s'].shape[1]
		
		## TODO Randomize split

		#Training Data
		self.train_cat=tmp['train_cat_s'][:,:(2*size/3)]
		self.train_left=tmp['train_left_s'][:,:(2*size/3)]
		self.train_right=tmp['train_right_s'][:,:(2*size/3)]


		#Validation Data
		self.val_cat=tmp['train_cat_s'][:,(2*size/3):]
		self.val_left=tmp['train_left_s'][:,(2*size/3):]
		self.val_right=tmp['train_right_s'][:,(2*size/3):]

		#Test Data
		self.test_cat=tmp['test_left_s']
		self.test_left=tmp['test_left_s']
		self.test_right=tmp['test_right_s']

		print "OK"
				
		#Show image
		#plt.imshow(sp.reshape(test_left_s[:,30],(24,24)))
		#plt.show()

	def normalizeData(self, data):

		norm_data = sp.zeros((data.shape))

		for i in range(data.shape[1]):
			image = data[:,i]

			mean = sp.mean(image)
			var = sp.std(image)

			image = (image - mean)/var
		
			## TODO Check with assistants if mean of -2.671474153e-16 instead of 0 is OK.

			norm_data[:,i] = image

		return norm_data

	def normalizeDataset(self):

		print "Normalizing training set...",
		self.train_left = self.normalizeData(self.train_left)
		self.train_right = self.normalizeData(self.train_right)
		print "OK"



