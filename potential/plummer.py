import numpy as np
from .potential import potential


class plummer(potential):
    def __init__(self, M_plummer, r_plummer):
        self.M_plummer = M_plummer
        self.r_plummer = r_plummer

    def potential(self, r):
        return -self.M_plummer / np.sqrt(self.r_plummer ** 2 + r ** 2)

    def pot_zero(self):
        return -self.M_plummer / self.r_plummer
