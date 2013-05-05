import scipy as sp
import numpy as np
from pprint import pprint 

class MLP:

	def __init__(self, H1, H2, dimension):
		self.H1 = H1
		self.H2 = H2
		self.dimension = dimension
		
		self.w1l = sp.random.normal(0, 1.0/(dimension), (H1, (dimension+1)))
		self.w1r = sp.random.normal(0, 1.0/(dimension), (H1, (dimension+1)))
		self.w2l = sp.random.normal(0, 1.0/H1, (H2, H1+1))
		self.w2lr = sp.random.normal(0, 1.0/(2*H1), (H2, (2*H1)+1))
		self.w2r = sp.random.normal(0, 1.0/H1, (H2, H1+1))
		self.w3 = sp.random.normal(0, 1.0/H2, (1, H2+1))
		
	def forward_pass(self, xL, xR):
		# First Layer
		bs = sp.ones((1,xL.shape[0]), dtype=float)
		
		xLb = sp.vstack([xL, bs])
		xRb = sp.vstack([xR, bs])

		a1L = sp.dot(self.w1l, xLb)
		a1R = sp.dot(self.w1r, xRb)
		
		z1L = sp.tanh(a1L)
		z1R = sp.tanh(a1R)
		
		# Second Layer		
		z1Lb = sp.vstack([z1L, bs])
		z1LRb = sp.vstack([z1L, z1R, bs])
		z1Rb = sp.vstack([z1R, bs])
		
		a2L = sp.dot(self.w2l, z1Lb)
		a2LR = sp.dot(self.w2lr, z1LRb)
		a2R = sp.dot(self.w2r, z1Rb)
		
		z2 = a2LR*self.sigmoid(a2L)*self.sigmoid(a2R)
		
		# Third Layer
		z2b = sp.vstack([z2, bs])
		
		a3 = sp.dot(self.w3, z2b)
		return a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, z2b
		
	def backward_pass(self, a1L, a1R, a2L, a2LR, a2R, a3, z1Lb, z1LRb, z1Rb, z2b, xL, xR, t):
	 	
	 	# Third Layer
	 	r3=	-t*self.sigmoid(-t*a3)
	 	grad3=sp.dot(r3,z2b.T);
	 	
	 	# Second Layer
	 	r3w3T = sp.dot(self.w3[:,:-1].T, r3)

	 	r2L=r3w3T*self.sigmoid(a2R)*self.divsigmoid(a2L)
	 	r2R=r3w3T*self.sigmoid(a2L)*self.divsigmoid(a2R)
	 	r2LR=r3w3T*self.sigmoid(a2L)*self.sigmoid(a2R)
	 	
	 	grad2L = sp.dot(r2L, z1Lb.T)
	 	grad2LR = sp.dot(r2LR, z1LRb.T)
	 	grad2R = sp.dot(r2R, z1Rb.T)
	 	
	 	# First Layer
	 	r1L = sp.power(1.0/sp.cosh(a1L),2)*(sp.dot(self.w2l[:,:-1].T, r2L)+sp.dot(self.w2lr[:,:self.H1].T, r2LR))
	 	r1R = sp.power(1.0/sp.cosh(a1R),2)*(sp.dot(self.w2r[:,:-1].T, r2R)+sp.dot(self.w2lr[:,self.H1:-1].T, r2LR))
	 	
	 	grad1L = sp.dot(r1L, xL.T)
		grad1R = sp.dot(r1R, xR.T)
		
		return grad3, grad2L, grad2LR, grad2R, grad1L, grad1R
	 	
	 	
	def sigmoid(self, x) : 
		return 1.0/(1.0+sp.exp(-x))
	def divsigmoid(self, x) :
		return sp.exp(-x)/sp.power((1.0+sp.exp(-x)),2)
	
	
	 	
	
