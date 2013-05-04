import scipy as sp


class Gradient:

	def __init__(self, nu, mu):
		"""
		nu: learning rate
		mu: momentum
		"""
		self.nu = nu
		self.mu = mu
		self.delta_w_old = None

	def descend(self, W, gradients):

		if self.delta_w_old == None:
			self.delta_w_old = sp.zeros(W.shape)


		updateW(W, gradients)

		
	def updateW(w_old, gradients)
			delta_w_new = -self.nu*(1-self.mu)*prime(w_new)+self.mu*self.delta_w_old
			w_new = w_old + delta_w_new
		self.delta_w_old = delta_w_new
		return (w_new)
 

