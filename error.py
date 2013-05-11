import scipy as sp

class Error:
	def __init__(self):
		pass

	def logistic_error(self, result, expected):
		return sp.log(1+sp.exp(-expected*result))

	def total_error(self, result, expected):
		total = 0

		for i in range(result.shape[0]):
			total += self.logistic_error(result[i], expected[i])
		total /= result.shape[0]

		return total

e = Error()
print e.total_error(sp.array([1]), sp.array([1]))