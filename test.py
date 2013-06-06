from data import Data
from mlp import MLP
from error import Error
import matplotlib.pyplot as plt
import scipy as sp
from scipy import linalg as LA
import matplotlib.cm as cm
#from pprint import pprint 


class Test:

	def __init__(self, mlp, data, k):
		self.mlp = mlp
		self.data = data
		self.k = k
		self.error = Error()

	def classify(self) :
		self.test_val, self.test_classif = self.mlp.classify(self.data.test_left, self.data.test_right)
		test_error, misclassified_test = self.error.norm_total_error(self.test_val, self.data.test_cat, self.k)
		print "TEST ERROR :"+str(test_error)
		print "TEST MISCLASSIFICATION RATIO :"+str(misclassified_test)

	def examples(self) :
		if self.k == 2 :
			misclass = self.test_val * self.data.test_cat
			print misclass
			print misclass[misclass < 0]
			large_pos = sp.argmax(misclass)
			print "Largest positive t*a : "+str(max(misclass[misclass > 0]))
			print "Category : "+str(self.data.test_cat[:,large_pos])
			print "Found : "+str(self.test_classif[:,large_pos])
			large_neg = sp.argmin(misclass)
			print "Largest negative t*a : "+str(min(misclass[misclass < 0]))
			print "Category : "+str(self.data.test_cat[:,large_neg])
			print "Found : "+str(self.test_classif[:,large_neg])
			misclass_neg = misclass
			misclass_neg[misclass_neg>0] = -float("inf")
			close_0 = sp.argmax(misclass_neg)
			print "Closest to zero negative t*a : "+str(max(misclass_neg[misclass_neg < 0]))
			print "Category : "+str(self.data.test_cat[:,close_0])
			print "Found : "+str(self.test_classif[:,close_0])

			#Show image
			best_fig = plt.figure()
			ax5 = best_fig.add_subplot(111)
			ax5.imshow(sp.reshape(self.data.test_left[:,large_pos],(24,24)))
			worst_fig = plt.figure()
			ax3 = worst_fig.add_subplot(111)
			ax3.imshow(sp.reshape(self.data.test_left[:,large_neg],(24,24)))
			almost_fig = plt.figure()
			ax4 = almost_fig.add_subplot(111)
			ax4.imshow(sp.reshape(self.data.test_left[:,close_0],(24,24)))
			plt.show()

		else :
			print self.test_val
			err = Error()
			maxerr = -float("inf")
			minerr = float("inf")
			besterr = float("inf")
			for i in range(self.test_val.shape[1]) :
				error = 0.5*sp.sum(LA.norm(self.test_val[:,i]-self.data.test_cat[:,i])**2)
				#print self.test_classif[i], self.data.test_cat[:,i].argmax() 
				if self.test_classif[i] != self.data.test_cat[:,i].argmax() :
					if error > maxerr :
						maxerr = error
						maxindex = i
					if error < minerr :
						minerr = error
						minindex = i
				else :
					if error < besterr :
						besterr = error
						bestindex = i

			print "Best : "+str(besterr)
			print "Category : "+str(self.data.test_cat[:,bestindex].argmax())
			print "Found : "+str(self.test_classif[bestindex])

			print "Largest negative error : "+str(maxerr)
			print "Category : "+str(self.data.test_cat[:,maxindex].argmax())
			print "Found : "+str(self.test_classif[maxindex])

			print "Smallest negative error : "+str(minerr)
			print "Category : "+str(self.data.test_cat[:,minindex].argmax())
			print "Found : "+str(self.test_classif[minindex])


			#Show image
			
			best_fig = plt.figure()
			ax5 = best_fig.add_subplot(111)
			ax5.imshow(sp.reshape(self.data.test_left[:,bestindex],(24,24)), cmap=plt.gray())
			worst_fig = plt.figure()
			ax3 = worst_fig.add_subplot(111)
			ax3.imshow(sp.reshape(self.data.test_left[:,maxindex],(24,24)), cmap=plt.gray())
			almost_fig = plt.figure()
			ax4 = almost_fig.add_subplot(111)
			ax4.imshow(sp.reshape(self.data.test_left[:,minindex],(24,24)), cmap=plt.gray())
			plt.show()

	def confusion_matrix(self) :
		if self.k == 2:
			confusion_matrix = sp.zeros(2)
		else :
			confusion_matrix = sp.zeros((self.k, self.k))
			for i in range(self.test_classif.shape[0]) :
				confusion_matrix[self.test_classif[i]][self.data.test_cat[:,i].argmax()] += 1
			print confusion_matrix

			conf = plt.figure()
			ax = conf.add_subplot(111)
			ax.imshow(confusion_matrix, cmap=cm.get_cmap(name='gray_r'), interpolation='nearest')
			plt.show()
			


