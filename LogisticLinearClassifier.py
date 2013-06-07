import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class LogisticLinearClassifier:

	def __init__(self, nu, mu, dimension, k, data):
		self.nu = nu
		self.mu = mu
		self.data = data
		self.w = sp.random.normal(0, 1.0/sp.sqrt(2*dimension+1), ((2*dimension+1),k))
		self.k = k

	def train(self, NB_EPOCH):

		"""
			Minimize sigma(yi) - ti
		"""

		

		x = sp.vstack([self.data.train_left,self.data.train_right, sp.ones(self.data.train_left.shape[1])]).T
		delta_w_old = sp.zeros(x.shape[1])
		

		err_train = sp.zeros(NB_EPOCH)
		miss_train = sp.zeros(NB_EPOCH)
		err_val = sp.zeros(NB_EPOCH)
		miss_val = sp.zeros(NB_EPOCH)

		for epoch in range(NB_EPOCH):
			#self.data.shuffleData()
			x = sp.vstack([self.data.train_left,self.data.train_right, sp.ones(self.data.train_left.shape[1])]).T
			delta_w_old = self.descent(x, self.data.train_cat, delta_w_old)
			
			res, classes = self.classify(self.data.val_left, self.data.val_right)
			error, misclassified = self.error(res.T, self.data.val_cat)

			err_val[epoch] = error/(self.data.val_left.shape[1]*1.0)
			miss_val[epoch] = misclassified/(self.data.val_left.shape[1]*1.0)

			print epoch, "Logistic error:", error/self.data.val_left.shape[1], "misclassified:", misclassified
			res, _ = self.classify(self.data.train_left, self.data.train_right)
			err, miss = self.error(res.T, self.data.train_cat)
			err_train[epoch] = err/(self.data.train_left.shape[1]*1.0)
			miss_train[epoch] = miss/(self.data.train_left.shape[1]*1.0)
			"""
			if epoch > 1:
				if err_val[epoch] > err_val[epoch-1]:
					print "overfitting"
					break
			"""
		
		

		return err_train, miss_train, err_val, miss_val


	def classify(self, xL, xR):

		x = sp.vstack([xL,xR, sp.ones(xR.shape[1])]).T

		tmp = sp.dot(x, self.w)

		return tmp, sp.argmax(tmp, axis=1)

	def error(self, results, expected):	
		
		err = 0
		for i in range(results.shape[1]):

			err += self.lsexp(results[:,i]) - sp.dot(expected[:,i],results[:,i])

		misclassified = sp.sum(sp.argmax(results,axis=0) != sp.argmax(expected, axis=0))

		return err, misclassified


	def normalized_error(self, error, size):
		return error/size

	def lsexp(self, array):
		return sp.log(sp.sum(sp.exp(array), 0))
		


	def descent(self, x, cat, delta_w_old):
		for i in range(x.shape[1]):

			x_ = sp.array([x[i,:]])
			cat_ = sp.array([cat[:,i]]).T
			
			y = sp.dot(self.w.T, x_.T)
			sigma = sp.exp(y - self.lsexp(y))
			
			gradients = sp.dot((sigma - cat_), x_)

			self.w, delta_w_old = self.updateW(self.w, gradients, delta_w_old)
		

		return delta_w_old

	def updateW(self, w_old, gradients, delta_w_old):

		delta_w_new = -self.nu*(1-self.mu)*gradients+self.mu*delta_w_old

		w_new = w_old + delta_w_new.T

		return w_new, delta_w_new

	def confusion_matrix(self, result, expected) :
		confusion_matrix = sp.zeros((self.k, self.k))
		for i in range(result.shape[0]) :
			confusion_matrix[result[i]][expected[i]] += 1
		print confusion_matrix

		conf = plt.figure()
		ax = conf.add_subplot(111)
		ax.imshow(confusion_matrix, cmap=cm.get_cmap(name='gray_r'), interpolation='nearest')
