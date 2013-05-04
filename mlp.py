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
		bs = sp.ones((1,xL.shape[1]), dtype=float)
		
		xLb = sp.vstack([xL, bs])
		xRb = sp.vstack([xR, bs])

		a1L = sp.dot(self.w1l, xLb)
		a1R = sp.dot(self.w1r, xRb)
		
		z1L = sp.tanh(a1L)
		z1R = sp.tanh(a1R)
		
		#pprint(z1L)
		
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
		
		print a3.shape
		pprint(a3)
		
	def backward_pass(self, z, a, x_L, x_R, t):
	 	pass
	 	
	def sigmoid(self, x) : 
		return 1/(1+sp.exp(-x))
	 	
	
	 	
	
