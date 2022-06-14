import numpy as np
from .potential import potential
from scipy import interpolate
import subprocess


class nfw(potential):
    """
    Navarro-Frenk-White potential
    This can be initialized either in terms of scale density and scale radius
    or in terms of virial mass, virial radius, and concentration
    """

    def __init__(self, rhos=None, rs=None, mvir=None, rvir=None, c=None, Msun=False, interp=False, h0=0.67):
        self.t_cur = None
        self.ttab = None
        self.ztab = None
        self.mvirtab = None
        self.conctab = None
        self.rvirtab = None
        self.mvirf = None
        self.concf = None
        self.rvirf = None
        self.mvir = None
        self.rhos = None
        self.rvir = None
        self.c = None
        self.rs = None
        self.h0 = h0  # H_0/(100 km/s/Mpc)
        self.interp = interp
        if (rhos is not None) and (rs is not None):
            self.rhos = rhos
            self.rs = rs
        elif (mvir is not None) and (rvir is not None) and (c is not None):
            if Msun:
                self.mvir = mvir / 24393036.0465
            else:
                self.mvir = mvir
            self.rvir = rvir
            self.c = c
            self.rs = self.rvir / self.c
        elif (mvir is not None):
            if Msun:
                self.mvir = mvir / 24393036.0465
            else:
                self.mvir = mvir
            self.evaluate_at_time("now")
        else:
            raise ValueError("Not enough information to initialize NFW potential")

    def evaluate_at_time(self, t):
        if t == self.t_cur: return
        if self.ttab is None:
            subprocess.call([r'/Users/users/gronnow/src/evolzhao',
                             str(self.mvir * 24393036.0465)])  # Run external programme to generate table for given mass
            self.ttab, self.ztab, self.mvirtab, self.conctab, self.rvirtab = np.loadtxt("haloevol.dat", skiprows=1,
                                                                                        unpack=True)
            self.ttab /= self.h0  # time in yr
            self.mvirtab /= 24393036.0465 * self.h0  # Virial mass in code mass units
            self.rvirtab *= 1000.0 / self.h0  # Virial radius in kpc
            if self.interp:
                self.mvirf = interpolate.interp1d(self.ttab, self.mvirtab)
                self.rvirf = interpolate.interp1d(self.ttab, self.rvirtab)
                self.concf = interpolate.interp1d(self.ttab, self.conctab)

        if t == "now":
            self.mvir = self.mvirtab[0]
            self.c = self.conctab[0]
            self.rvir = self.rvirtab[0]
        else:
            if self.interp:
                self.mvir = self.mvirf(t)
                self.c = self.concf(t)
                self.rvir = self.rvirf(t)
            else:
                idx = (np.abs(t - self.ttab)).argmin()
                self.mvir = self.mvirtab[idx]
                self.c = self.conctab[idx]
                self.rvir = self.rvirtab[idx]

        self.rs = self.rvir / self.c
        self.rhos = None
        self.t_cur = t

    def now(self):
        return self.ttab[0]

    def set_params(self, rhos, rs):
        self.rhos = rhos
        self.rs = rs

    def set_params(self, mvir, rvir, c):
        self.mvir = mvir
        self.rvir = rvir
        self.c = c
        self.rs = self.rvir / self.c

    def calc_params(self, rvir=None):
        if self.rhos is None:
            self.rhos = self.mvir / (4.0 * np.pi * self.rs ** 3 * (np.log(1.0 + self.c) - self.c / (1.0 + self.c)))
        elif (self.mvir is None) and (rvir is not None):
            self.rvir = rvir
            self.c = rvir / self.rs
            self.mvir = 4.0 * np.pi * self.rhos * self.rs ** 3 * (np.log(1.0 + self.c) - self.c / (1.0 + self.c))

    def potential(self, r, t=None):
        if t is not None:
            self.evaluate_at_time(t)
        return self.pot_zero() * self.rs * np.log(1.0 + r / self.rs) / r

    def pot_zero(self, t=None):
        if t is not None:
            self.evaluate_at_time(t)
            potzero = -self.mvir / (self.rs * (np.log(1.0 + self.c) - self.c / (1.0 + self.c)))
        elif self.rhos is None:
            potzero = -self.mvir / (self.rs * (np.log(1.0 + self.c) - self.c / (1.0 + self.c)))
        else:
            potzero = -4.0 * np.pi * self.rhos * self.rs ** 2

        return potzero

    def acceleration(self, r, t=None):
        if t is not None:
            self.evaluate_at_time(t)
        return self.pot_zero() * (self.rs / r ** 2) * (np.log(1.0 + r / self.rs) - (r / (self.rs + r)))

    def mass_enclosed(self, r, t=None):
        if t is not None:
            self.evaluate_at_time(t)
        return -self.pot_zero() * self.rs * (np.log(1.0 + r / self.rs) + self.rs / (self.rs + r) - 1.0)
