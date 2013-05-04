import scipy as sp
import numpy as np
from pprint import pprint 

class MLP:

	def __init__(self, H1, H2, dimension):
		self.H1 = H1
		self.H2 = H2
		self.dimension = dimension
		
		self.w1l = np.matrix(sp.random.normal(0, 1.0/(dimension), (H1, (dimension+1))))
		self.w1r = np.matrix(sp.random.normal(0, 1.0/(dimension), (H1, (dimension+1))))
		self.w2l = np.matrix(sp.random.normal(0, 1.0/H1, (H2, H1+1)))
		self.w2lr = np.matrix(sp.random.normal(0, 1.0/(2*H1), (H2, (2*H1)+1)))
		self.w2r = np.matrix(sp.random.normal(0, 1.0/H1, (H2, H1+1)))
		self.w3 = np.matrix(sp.random.normal(0, 1.0/H2, (1, H2+1)))
		
		pprint(self.w1l)

	def forward_pass(self, xL, xR):
		xL=np.matrix(xL)
		xR=np.matrix(xR)
			
		# First Layer
		bs = sp.ones((1,xL.shape[1]), dtype=float)
		
		xLb = sp.vstack([xL, bs])
		xRb = sp.vstack([xR, bs])

		a1L = self.w1l * xLb
		a1R = self.w1r * xRb
		
		z1L = sp.tanh(a1L)
		z1R = sp.tanh(a1R)
		
		#pprint(z1L)
		
		# Second Layer		
		z1Lb = sp.vstack([z1L, bs])
		z1LRb = sp.vstack([z1L, z1R, bs])
		z1Rb = sp.vstack([z1R, bs])
		
		a2L = self.w2l * z1Lb
		a2LR = self.w2lr * z1LRb
		a2R = self.w2r* z1Rb
		
		z2 = sp.multiply(a2LR,sp.multiply((self.sigmoid(a2L)),(self.sigmoid(a2R))))
		# Third Layer
		z2b = sp.vstack([z2, bs])
		
		a3 = self.w3 * z2b
		
		print a3.shape
		pprint(a3)
		
	def backward_pass(self, z, a, x_L, x_R, t):
	 	pass
	 	
	def sigmoid(self, x) : 
		return 1/(1+sp.exp(-x))
	 	
	
	 	
	
