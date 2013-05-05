import scipy as sp


class Gradient:

	def __init__(self, nu, mu):
		"""
		nu: learning rate
		mu: momentum
		"""
		self.nu = nu
		self.mu = mu

		#
		self.delta_w1l_old = None
		self.delta_w1r_old = None
		self.delta_w2l_old = None
		self.delta_w2r_old = None
		self.delta_w2lr_old = None
		self.delta_w3_old = None

	def descend(self,  w1l, w1r, w2l, w2r, w2lr, w3, grad3, grad2L, grad2LR, grad2R, grad1L, grad1R):

		if self.delta_w1l_old == None:
			self.delta_w1l_old = sp.zeros(w1l.shape)
			self.delta_w1r_old = sp.zeros(w1r.shape)
			self.delta_w2l_old = sp.zeros(w2l.shape)
			self.delta_w2r_old = sp.zeros(w2r.shape)
			self.delta_w2lr_old = sp.zeros(w2lr.shape)
			self.delta_w3_old = sp.zeros(w3.shape)

		new_w1l, self.delta_w1l_old = self.updateW(w1l, grad1L, self.delta_w1l_old)
		new_w1r, self.delta_w1r_old = self.updateW(w1r, grad1R, self.delta_w1r_old)
		new_w2l, self.delta_w2l_old = self.updateW(w2l, grad2L, self.delta_w2l_old) 
		new_w2r, self.delta_w2r_old = self.updateW(w2r, grad2R, self.delta_w2r_old ) 
		new_w2lr, self.delta_w2lr_old = self.updateW(w2lr, grad2LR, self.delta_w2lr_old)
		new_w3, self.delta_w3_old = self.updateW(w3, grad3, self.delta_w3_old)

		return new_w1l, new_w1r, new_w2l, new_w2r, new_w2lr, new_w3

	def updateW(self, w_old, gradients, delta_w_old):
		delta_w_new = -self.nu*(1-self.mu)*gradients+self.mu*delta_w_old
		w_new = w_old + delta_w_new
		return w_new, delta_w_new


