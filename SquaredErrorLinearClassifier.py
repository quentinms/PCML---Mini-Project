import scipy as sp
import scipy.linalg as LA

class SquaredErrorLinearClassifier:

	def __init__(self, v, k, data):
		self.v = v
		self.k = k
		self.data = data
		self.w = None

	def train(self):

		"""
			Solve (Phi.T*Phi + v I)w = Phi.T t (see p.115)

			#TODO: Add constant feature

		"""

		x = sp.vstack([self.data.train_left,self.data.train_right, sp.ones(self.data.train_left.shape[1])]).T

		A = sp.dot(x.T, x) + sp.dot(self.v, sp.eye(x.shape[1]))
		b = sp.dot(x.T, self.data.train_cat.T)
		
		self.w, _, _, _ = LA.lstsq(A,b)


	def classify(self):

		x = sp.vstack([self.data.val_left,self.data.val_right, sp.ones(self.data.val_left.shape[1])]).T

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

		print sq_error, tikhonov

		misclassified = sp.sum(sp.argmax(results,axis=0)!=sp.argmax(expected, axis=0))

		return error, misclassified
