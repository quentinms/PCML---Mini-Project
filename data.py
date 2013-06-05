import scipy as sp
import scipy.io as spio
from matplotlib import pyplot as plt

class Data:

	def __init__(self, k, train_size, validation_size):
		self.k = k
		self.train_size = train_size
		self.validation_size = validation_size

	def importDataFromMat(self):
		print "Importing data ...",

		if self.k == 2 :
			tmp = spio.loadmat('miniproject_data/norb_binary.mat')
		else :
			tmp = spio.loadmat('miniproject_data/norb_5class.mat')

		
		size=tmp['train_cat_s'].shape[1]
		print size

		#Randomize indices
		sp.random.seed(1)
		train_set_indices=sp.random.choice(size, 2*size/3, False)
		complete_set_indices=sp.arange(size)
		val_set_indices=sp.setdiff1d(complete_set_indices,train_set_indices);
		if (self.train_size > 0) & (self.train_size < 2*size/3) :
			train_set_indices=sp.random.choice(train_set_indices, self.train_size, False)
		if (self.validation_size > 0) & (self.validation_size < size/3) :
			val_set_indices=sp.random.choice(val_set_indices, self.validation_size, False)

		#Training Data
		self.train_cat=sp.array(tmp['train_cat_s'][:,train_set_indices], dtype='int8')
		self.train_left=sp.array(tmp['train_left_s'][:,train_set_indices],dtype=float)
		self.train_right=sp.array(tmp['train_right_s'][:,train_set_indices],dtype=float)

		#Validation Data
		self.val_cat=sp.array(tmp['train_cat_s'][:,val_set_indices], dtype='int8')
		self.val_left=sp.array(tmp['train_left_s'][:,val_set_indices],dtype=float)
		self.val_right=sp.array(tmp['train_right_s'][:,val_set_indices],dtype=float)

		#Test Data
		self.test_cat=sp.array(tmp['test_cat_s'], dtype='int8')
		self.test_left=sp.array(tmp['test_left_s'], dtype=float)
		self.test_right=sp.array(tmp['test_right_s'], dtype=float)

		print "OK"
				
		#Show image
		#plt.imshow(sp.reshape(self.test_left[:,33],(24,24)))
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
		print "OK"

		print "Normalizing validation set...",
		self.val_left = self.normalizeDataset(self.val_left)
		self.val_right = self.normalizeDataset(self.val_right)
		print "OK"

		print "Normalizing test set...",
		self.test_left = self.normalizeDataset(self.test_left)
		self.test_right = self.normalizeDataset(self.test_right)
		print "OK"


		print "Normalizing categories' labels...",

		if self.k == 2:
			categories = sp.unique(self.train_cat)
			self.train_cat[self.train_cat == categories[0]] = -1
			self.train_cat[self.train_cat == categories[1]] = 1
			self.val_cat[self.val_cat == categories[0]] = -1
			self.val_cat[self.val_cat == categories[1]] = 1	
			self.test_cat[self.test_cat == categories[0]] = -1
			self.test_cat[self.test_cat == categories[1]] = 1

		else :

			self.val_cat = self.buildCatMatrix(self.val_cat)
			self.train_cat = self.buildCatMatrix(self.train_cat)
			self.test_cat = self.buildCatMatrix(self.test_cat)

		print "OK"

	def buildCatMatrix(self, cat) :
		mat_cat = sp.zeros((self.k,cat.shape[1]))
		for i in range(cat.shape[1]) :
			for j in range(self.k) :
				if cat[0,i]==j :
					mat_cat[j,i] = 1

		return mat_cat

	def shuffleData(self):

		print "Shuffling training set...",
		data_length = self.train_cat.shape[1]
		rand_index = sp.random.permutation(data_length)

		self.train_cat = self.train_cat[:,rand_index]
		self.train_left = self.train_left[:,rand_index]
		self.train_right = self.train_right[:,rand_index]
		print "OK"
		
