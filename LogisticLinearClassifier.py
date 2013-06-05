import scipy as sp
import matplotlib.pyplot as plt

class LogisticLinearClassifier:

	def __init__(self, nu, mu, dimension, k, data):
		self.nu = nu
		self.mu = mu
		self.data = data
		self.w = sp.random.normal(0, 1.0/sp.sqrt(2*dimension+1), ((2*dimension+1),k))

	def train(self):

		"""
			Minimize sigma(yi) - ti
		"""

		delta_w_old = 0

		x = sp.vstack([self.data.train_left,self.data.train_right, sp.ones(self.data.train_left.shape[1])]).T


		NB_EPOCH = 150
		err = sp.zeros(NB_EPOCH)
		miss = sp.zeros(NB_EPOCH)

		for epoch in range(NB_EPOCH):
			self.data.shuffleData()
			x = sp.vstack([self.data.train_left,self.data.train_right, sp.ones(self.data.train_left.shape[1])]).T
			delta_w_old = self.descent(x, self.data.train_cat, delta_w_old)

			res, classes = self.classify(self.data.val_left, self.data.val_right)
			error, misclassified = self.error(res.T, self.data.val_cat)

			err[epoch] = error
			miss[epoch] = misclassified

			print epoch, "Logistic error:", error/self.data.train_left.shape[1], "misclassified:", misclassified
		
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(err, label='logistic error')
		fig2 = plt.figure()
		ax2 = fig2.add_subplot(111)
		ax2.plot(miss, label='misclassified')

		plt.show()


	def classify(self, xL, xR):

		x = sp.vstack([xL,xR, sp.ones(xR.shape[1])]).T

		tmp = sp.dot(x, self.w)

		return tmp, sp.argmax(tmp, axis=1)

	def error(self, results, expected):	
		
		#err = sp.sum(sp.tile(sp.array([self.lsexp(results)]).T,5).T-(expected*results))
		err = 0
		for i in range(results.shape[1]):

			if i == -1:
				print sp.misc.logsumexp(results[:,i]).shape
				print expected[:,i].shape
				print results[:,i].shape
				print sp.dot(expected[:,i],results[:,i]).shape

			err += sp.misc.logsumexp(results[:,i]) - sp.dot(expected[:,i],results[:,i])


		misclassified = sp.sum(sp.argmax(results,axis=0) != sp.argmax(expected, axis=0))

		return err, misclassified


	def normalized_error(self, error, size):
		return error/size

	def lsexp(self, array):
		# <=> sp.log(sp.sum(sp.exp(array)))
		return sp.log(sp.sum(sp.exp(array), 0))
		#return sp.misc.logsumexp(array)

	def gradients(self):
		pass

	def descent(self, x, cat, delta_w_old):
		for i in range(x.shape[1]):

			x_ = sp.array([x[i,:]])
			cat_ = sp.array([cat[:,i]]).T

			sigma = 1 / (1 + sp.exp(-sp.dot(self.w.T, x_.T)))

			gradients = sp.dot((sigma - cat_) , x_)
			self.w, delta_w_old = self.updateW(self.w, gradients, delta_w_old)

		return delta_w_old

	def updateW(self, w_old, gradients, delta_w_old):

		delta_w_new = -self.nu*(1-self.mu)*gradients+self.mu*delta_w_old

		w_new = w_old + delta_w_new.T

		

		return w_new, delta_w_new


