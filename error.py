import scipy as sp
from scipy import linalg as LA

class Error:
	def __init__(self):
		pass

	def logistic_error(self, result, expected):
		return sp.log(1+sp.exp(-expected*result))
		#return sp.absolute(result-expected)

	def total_error(self, result, expected, k):
		error = 0
		tmp = - result * expected
		

		if k==2 :
			#Negative error
			error += sp.sum(sp.log1p(sp.exp(tmp[tmp<0])))
			#Positive error
			error += sp.sum(tmp[tmp>=0]+sp.log(1+sp.exp(tmp[tmp>=0])))
			print result
			misclassified = sp.sum(sp.absolute(sp.sign(result)-expected))/2
		else :
			error += 0.5*sp.sum(LA.norm(result-expected)**2)
			misclassified = sp.sum(sp.argmax(result,axis=0)!=sp.argmax(expected, axis=0))
		#error /= result.shape[1]
		misclassified /= result.shape[1]*1.0

		return error, misclassified

	def norm_total_error(self, result, expected, k):
		error, misclassified = self.total_error(result, expected, k)
		error /= result.shape[1]

		return error, misclassified