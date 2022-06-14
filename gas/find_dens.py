import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import dblquad
from ..hydrostatic_prof import hydrostatic_prof


class find_dens:

    def __init__(self, pot, T_cold, T_hot, n_hot):
        self.gasprof = hydrostatic_prof(pot, 1.0, T_cold)  # n0 set to 1 initially so that it can be easily scaled
        self.sfr_fac = 12.59  # Bacchinni et al. 2020
        self.sfr_exp = 2.03  # Bacchinni et al. 2020
        self.T_hot = T_hot
        self.n_hot = n_hot
        self.n_eq = n_hot * T_hot / T_cold
        self.rgas = 0.0

    def central_density(self, cylrad_sfr:float, sfr:float) -> tuple:
        """
        Calculate the scale density that leads to a star formation rate of sfr within a cylindrical radius of cylrad_sfr
        :param cylrad_sfr: cylindrical projected radius within which SFR was derived in kpc
        :param sfr: total SFR within cylrad_sfr in Msun/yr
        :return: scale density [amu/cm^3] and radius [kpc] where the hot and cold gas is in pressure equilibrium
        """
        try:
            res = root_scalar(self.rho_central_cyl, args=(cylrad_sfr, sfr), bracket=(1.01 * self.n_eq, 20.0),
                              x0=100.0 * self.n_eq)
        except ValueError:
            print("ValueError in rho_central_cyl")
            return -1, -1
        rho0 = res.root
        mu = self.gasprof.mu()
        rgas = self.get_rgas(rho0 / mu)
        return res.root, rgas

    def rho_central_cyl(self, rho0trial:float, cylrad_sfr:float, sfr:float) -> float:
        """
        Calculates density from SFR and return difference from given trial density
        For use in root finding
        :param rho0trial: trial density in amu/cm^3
        :param cylrad_sfr: cylindrical radius within which SFR was derived in kpc
        :param sfr: total SFR within cylrad_sfr in Msun/yr
        :return: Calculated density minus rho0trial in amu/cm^3
        """
        facHe = 1.36  # Hydrogen mass to total mass conversion factor
        mu = self.gasprof.mu()
        self.rgas = self.get_rgas(rho0trial / mu)
        rhoH = dblquad(self.sfrHdens_cyl, 0, cylrad_sfr, lambda z: 0, self.ub)
        rho0 = facHe * (sfr / rhoH[0]) ** (1.0 / self.sfr_exp)
        return rho0 - rho0trial

    def ub(self, z:float) -> float:
        if z > self.rgas: return 0.0
        return np.sqrt(self.rgas ** 2 - z ** 2)

    def get_rgas(self, n0trial:float) -> float:
        P_hot = self.n_hot * self.T_hot
        try:
            res = root_scalar(lambda r: n0trial * self.gasprof.ndens(r) * self.gasprof.T - P_hot, bracket=(0.001, 10.0),
                              x0=0.1)
        except ValueError:
            print("ValueError in get_rgas")
            return -1
        return res.root

    def sfrHdens_cyl(self, z:float, cylrad:float) -> float:
        rho_convfac = 0.02439303604  # conversion factor Msun/pc**3 -> amu/cm**3
        r = np.sqrt(cylrad ** 2 + z ** 2)
        return 4.0 * np.pi * self.sfr_fac * rho_convfac ** self.sfr_exp * np.exp(
            -self.sfr_exp * self.gasprof.potentialdiff(r) / self.gasprof.cs()) * cylrad

    def mass(self, n0:float) -> tuple:
        rgas = self.get_rgas(n0)
        if rgas != -1:
            mass = self.gasprof.menc(rgas)
        else:
            mass = -1
        return rgas, mass
