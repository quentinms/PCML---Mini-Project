import scipy as sp
import scipy.linalg as LA

class SquaredErrorLinearClassifier:

	def __init__(self, v, k, data):
		self.v = v
		self.k = k
		self.data = data
		self.w = None

	def train(self, xL, xR, cat):

		"""
			Solve (Phi.T*Phi + v I)w = Phi.T t (see p.115)

		"""

		x = sp.vstack([xL,xR, sp.ones(xL.shape[1])]).T

		A = sp.dot(x.T, x) + sp.dot(self.v, sp.eye(x.shape[1]))
		b = sp.dot(x.T, cat.T)
		
		self.w, _, _, _ = LA.lstsq(A,b)


	def classify(self, xL, xR):

		x = sp.vstack([xL,xR, sp.ones(xR.shape[1])]).T

		tmp = sp.dot(x, self.w)

		return tmp, sp.argmax(tmp, axis=1)

	def error(self, results, expected):
		
		#Square Error
		sq_error = 0.5*sp.sum(LA.norm(results-expected)**2)

		#Tikhonov regularization
		w_norm = 0
		for i in range(0, self.k):
			w_norm += LA.norm(self.w[:,i])**2 

		
		tikhonov = self.v * w_norm

		error  = sq_error + tikhonov

		misclassified = sp.sum(sp.argmax(results,axis=0)!=sp.argmax(expected, axis=0))

		return error, misclassified

	def normalized_error(self, error, size):

		return error/size

	def crossvalidation(self, fold, v):
		#Split dataset in 10
		#Compute error for training set \ training set _ i
		#Average

		self.v = v

		error = 0

		n = self.data.train_cat.shape[1]
		all_indices = sp.arange(n)
		self.data.shuffleData()

		dataset_indices = sp.split(all_indices, fold)

		for i in range(fold):
			set_without_D_i_indices = sp.concatenate(dataset_indices[0:i]+dataset_indices[i+1:fold])
			
			print "-"*30+"train"+"-"*30
			self.train(self.data.train_left[:,set_without_D_i_indices], self.data.train_right[:,set_without_D_i_indices], self.data.train_cat[:,set_without_D_i_indices])
			print "-"*30+"classify"+"-"*30
			results, classes = self.classify(self.data.train_left[:,dataset_indices[i]], self.data.train_right[:,dataset_indices[i]])
			print "-"*30+"error"+"-"*30
			err,  _ = self.error(results, self.data.train_cat[:,dataset_indices[i]].T)

			error += fold/(n*1.0)*err

		error = error / fold

		return error


