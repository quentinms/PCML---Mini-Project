import scipy as sp
import scipy.linalg as LA

class SquaredErrorLinearClassifier:

	def __init__(self, v, k):
		self.v = v
		self.k = k
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



