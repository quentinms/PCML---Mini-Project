import scipy as sp
#from data import Data
#from pprint import pprint 
from error import Error

class MLP:

	def __init__(self, H1, H2, dimension, nu, mu, k, data):
		self.data = data

		self.H1 = H1
		self.H2 = H2
		self.dimension = dimension

		self.k = k;
		
		self.w1l = sp.random.normal(0, 1.0/sp.sqrt(dimension), (H1, (dimension+1)))
		self.w1r = sp.random.normal(0, 1.0/sp.sqrt(dimension), (H1, (dimension+1)))
		self.w2l = sp.random.normal(0, 1.0/sp.sqrt(H1), (H2, H1+1))
		self.w2lr = sp.random.normal(0, 1.0/sp.sqrt((2*H1)), (H2, (2*H1)+1))
		self.w2r = sp.random.normal(0, 1.0/sp.sqrt(H1), (H2, H1+1))
		if self.k==2 :
			self.w3 = sp.random.normal(0, 1.0/sp.sqrt(H2), (1, H2+1))
		else :
			self.w3 = sp.random.normal(0, 1.0/sp.sqrt(H2), (self.k, H2+1))
		"""
		nu: learning rate
		mu: momentum
		"""
		self.nu = nu
		self.mu = mu

		#
		self.delta_w1l_old = sp.zeros(self.w1l.shape)
		self.delta_w1r_old = sp.zeros(self.w1r.shape)
		self.delta_w2l_old = sp.zeros(self.w2l.shape)
		self.delta_w2r_old = sp.zeros(self.w2r.shape)
		self.delta_w2lr_old = sp.zeros(self.w2lr.shape)
		self.delta_w3_old = sp.zeros(self.w3.shape)
		
		

		
	def forward_pass(self, xL, xR):

		bs = sp.ones((1,xL.shape[1]), dtype=float)

		# First Layer
		xLb = sp.vstack([xL, bs])
		xRb = sp.vstack([xR, bs])

		a1L = sp.dot(self.w1l, xLb)
		a1R = sp.dot(self.w1r, xRb)
		
		z1L = sp.tanh(a1L)
		z1R = sp.tanh(a1R)
		#pprint(a1R)
		# Second Layer		
		z1Lb = sp.vstack([z1L, bs])
		z1LRb = sp.vstack([z1L, z1R, bs])
		z1Rb = sp.vstack([z1R, bs])

		a2L = sp.dot(self.w2l, z1Lb)
		a2LR = sp.dot(self.w2lr, z1LRb)
		a2R = sp.dot(self.w2r, z1Rb)
		#pprint(a2R)
		z2 = a2LR*self.sigmoid(a2L)*self.sigmoid(a2R)

		# Third Layer
		z2b = sp.vstack([z2, bs])
		a3 = sp.dot(self.w3, z2b)

		#pprint(a3)
		#pprint(self.data.val_cat)
		return a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, z2b, xLb, xRb
		
	def backward_pass(self, a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, z2b, xLb, xRb, t):
	 	
		# Third Layer
		if self.k == 2 :
			r3=	-t*self.sigmoid(-t*a3)
		else :
			#print a3
			#print t.T
			r3 = a3 - t.T

		grad3=sp.dot(r3,z2b.T)
		
		# Second Layer
		r3w3T = sp.dot(self.w3[:,:-1].T, r3)

		r2L=r3w3T*a2LR*self.sigmoid(a2R)*self.divsigmoid(a2L)
		r2R=r3w3T*a2LR*self.sigmoid(a2L)*self.divsigmoid(a2R)
		r2LR=r3w3T*self.sigmoid(a2L)*self.sigmoid(a2R)
		
		grad2L = sp.dot(r2L, z1Lb.T)
		grad2LR = sp.dot(r2LR, z1LRb.T)
		grad2R = sp.dot(r2R, z1Rb.T)
		
		# First Layer
		r1L = sp.power(1.0/sp.cosh(a1L),2)*(sp.dot(self.w2l[:,:-1].T, r2L)+sp.dot(self.w2lr[:,:self.H1].T, r2LR))
		r1R = sp.power(1.0/sp.cosh(a1R),2)*(sp.dot(self.w2r[:,:-1].T, r2R)+sp.dot(self.w2lr[:,self.H1:-1].T, r2LR))
		
		grad1L = sp.dot(r1L, xLb.T)
		grad1R = sp.dot(r1R, xRb.T)
		
		
		return grad3, grad2L, grad2LR, grad2R, grad1L, grad1R

	def descend(self, xL, xR, t):
		
		if self.k==2 :
			res = sp.zeros((1, xL.shape[1]))
		else :
			res = sp.zeros((self.k, xL.shape[1]))

		for i in range(1,xL.shape[1]) :
			a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, z2b, xLb, xRb = self.forward_pass(sp.array([xL[:,i]]).T, sp.array([xR[:,i]]).T)
			grad3, grad2L, grad2LR, grad2R, grad1L, grad1R = self.backward_pass(a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, z2b, xLb, xRb, sp.array([t[:,i]]));

			self.w1l, self.delta_w1l_old = self.updateW(self.w1l, grad1L, self.delta_w1l_old)
			self.w1r, self.delta_w1r_old = self.updateW(self.w1r, grad1R, self.delta_w1r_old)
			self.w2l, self.delta_w2l_old = self.updateW(self.w2l, grad2L, self.delta_w2l_old) 
			self.w2r, self.delta_w2r_old = self.updateW(self.w2r, grad2R, self.delta_w2r_old ) 
			self.w2lr, self.delta_w2lr_old = self.updateW(self.w2lr, grad2LR, self.delta_w2lr_old)
			self.w3, self.delta_w3_old = self.updateW(self.w3, grad3, self.delta_w3_old)

			if self.k==2 :
				res[0,i] = a3
			else :
				res[:,i] = a3.squeeze()

		return res

	def updateW(self, w_old, gradients, delta_w_old):
		delta_w_new = -self.nu*(1-self.mu)*gradients+self.mu*delta_w_old
		w_new = w_old + delta_w_new
		#if w_new.all() == w_old.all():
		#	print "Nope"
		#w_new = w_old - self.mu * gradients
		return w_new, delta_w_new


	def train(self):
		res = self.descend(self.data.train_left, self.data.train_right,self.data.train_cat)
		return res

	def classify(self):
		a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, z2b, xLb, xRb = self.forward_pass(self.data.val_left, self.data.val_right)
		for i in range(a3.shape[1]):
			if i % 10 == 0:
				if self.k == 2 :
					print sp.sign(a3[:,i]), (self.data.val_cat[0,i])
				else :	
					print sp.argmax(a3[:,i], axis=0), sp.argmax(self.data.val_cat[:,i])
		if self.k == 2 :
			classif = sp.sign(a3);
		else :
			classif = sp.argmax(a3,axis=0);
		return a3, classif
	 	
	def sigmoid(self, x) : 
		return 1.0/(1.0+sp.exp(-x))
	def divsigmoid(self, x) :
		return sp.exp(-x)/sp.power((1.0+sp.exp(-x)),2)

	def test_gradient(self):
		epsilon = 10**(-2)

		a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, z2b, xLb, xRb = self.forward_pass(self.data.val_left, self.data.val_right)
		grad3, grad2L, grad2LR, grad2R, grad1L, grad1R = self.backward_pass(a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, z2b, xLb, xRb, self.data.val_cat)
		print "grad3 shape : "+str(grad3.shape)
		print "w3 shape : "+str(self.w3.shape)
		print "w3 initial = "+str(self.w3[0,1])
		e=Error()
		self.w1l[0,8] += epsilon
		a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, z2b, xLb, xRb = self.forward_pass(self.data.val_left, self.data.val_right)
		e_plus = e.total_error(a3,self.data.val_cat)[0]
		print "E+ = "+str(e_plus)
		self.w1l[0,8] -= (2*epsilon)
		a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, z2b, xLb, xRb = self.forward_pass(self.data.val_left, self.data.val_right)
		e_minus = e.total_error(a3,self.data.val_cat)[0]
		print "E- = "+str(e_minus)
		self.w1l[0,8] += epsilon

		difference = (e_plus-e_minus)
		print 2*epsilon
		approx_grad = difference/(2*epsilon)
		grad=grad1L[0,8]

		print "Derivative = "+str(approx_grad)
		print "Gradient = "+str(grad1L[0,8])


		print "Difference "+str(approx_grad-grad)
