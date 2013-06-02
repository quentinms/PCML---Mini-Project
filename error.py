import scipy as sp

class Error:
	def __init__(self):
		pass

	def logistic_error(self, result, expected):
		return sp.log(1+sp.exp(-expected*result))
		#return sp.absolute(result-expected)

	def total_error(self, result, expected):
		error = 0
		tmp = - result * expected

		#Negative error
		error += sp.sum(sp.log1p(sp.exp(tmp[tmp<0])))
		#Positive error
		error += sp.sum(tmp[tmp>=0]+sp.log(1+sp.exp(tmp[tmp>=0])))
		error /= result.shape[1]

		misclassified = sp.sum(sp.absolute(sp.sign(result)-expected))/2


		return error, misclassified