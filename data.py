import scipy as sp
import scipy.io as spio
#from matplotlib import pyplot as plt

class Data:

	def __init__(self, fileName):
		self.fileName = fileName

	def importDataFromMat(self):
		tmp = spio.loadmat(self.fileName)

		print "Importing", self.fileName+"...",

		size=tmp['train_cat_s'].shape[1]

		#Randomize indices
		train_set_indices=sp.random.choice(size, (2*size/3), False)
		complete_set_indices=sp.arange(size)
		val_set_indices=sp.setdiff1d(complete_set_indices,train_set_indices);

		#Training Data
		self.train_cat=tmp['train_cat_s'][:,train_set_indices]
		self.train_left=tmp['train_left_s'][:,train_set_indices]
		self.train_right=tmp['train_right_s'][:,train_set_indices]

		#Validation Data
		self.val_cat=tmp['train_cat_s'][:,val_set_indices]
		self.val_left=tmp['train_left_s'][:,val_set_indices]
		self.val_right=tmp['train_right_s'][:,val_set_indices]

		#Test Data
		self.test_cat=tmp['test_left_s']
		self.test_left=tmp['test_left_s']
		self.test_right=tmp['test_right_s']

		print "OK"
				
		#Show image
		#plt.imshow(sp.reshape(test_left_s[:,30],(24,24)))
		#plt.show()

	def normalizeTrainingData(self, data):

		norm_data = sp.zeros((data.shape))
		self.data_mean = sp.zeros(data.shape[0])
		self.data_var = sp.zeros(data.shape[0])

		for i in range(data.shape[0]):
			pixel_i = data[i,:]

			mean = sp.mean(pixel_i)
			var = sp.std(pixel_i)

			pixel_i = (pixel_i - mean)/var
		
			## TODO Check with assistants if mean of -2.671474153e-16 instead of 0 is OK.

			norm_data[i,:] = pixel_i
			self.data_mean[i] = mean
			self.data_var[i] = var

		return norm_data

	def normalizeDataset(self, data):

		for i in range(data.shape[1]):
			data[:,i]=(data[:,i]-self.data_mean)/self.data_var

		return data

	def normalize(self):

		print "Normalizing training set...",
		self.train_left = self.normalizeTrainingData(self.train_left)
		self.train_right = self.normalizeTrainingData(self.train_right)

		print "Normalizing validation set...",
		self.val_left = self.normalizeDataset(self.val_left)
		self.val_right = self.normalizeDataset(self.val_right)
		print "OK"



