import numpy as np
from .potential import potential


class cored(potential):
    def __init__(self, rhos, rs):
        self.rhos = rhos
        self.rs = rs

    def potential(self, r):
        return self.pot_zero()*(-self.rs*(r+2*(r+self.rs)*np.log(self.rs)-2*(r+self.rs)*np.log(r+self.rs)))/(r*(r+self.rs))

    def pot_zero(self):
        return -2.0*np.pi*self.rhos*self.rs**2
