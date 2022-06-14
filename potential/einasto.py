import numpy as np
from .potential import potential


class einasto(potential):
    def __init__(self, rhos, rs, n):
        self.rhos = rhos
        self.rs = rs
        self.n = n

    def potential(self, r):
        return -(4.0*np.pi*self.rhos**self.rs**3*self.n*gamma(3*self.n)/r) * (1-inc_gamma(3*self.n,(r/self.rs)**(1/self.n))/gamma(3*self.n) + \
                                                                           (r/self.rs)*inc_gamma(2*self.n,(r/self.rs)**(1/self.n))/gamma(3*self.n))

    def pot_zero(self):
        return -4.0*np.pi*self.rhos*self.rs**2*self.n*gamma(2*self.n)

