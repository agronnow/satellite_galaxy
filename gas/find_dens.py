import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import dblquad
from ..hydrostatic_prof import hydrostatic_prof

class find_dens:

  def __init__(self,pot,T_cold,T_hot,n_hot):
    self.gasprof = hydrostatic_prof(pot,1.0,T_cold) #n0 set to 1 initially so that it can be easily scaled
    self.sfr_fac = 12.59 #Bacchinni et al. 2020
    self.sfr_exp = 2.03  #Bacchinni et al. 2020
    self.T_hot = T_hot
    self.n_hot = n_hot
    self.n_eq = n_hot*T_hot/T_cold

  def central_density(self,Rsfr,sfr):
     try:
       res=root_scalar(self.rho_central_cyl,args=(Rsfr,sfr),bracket=(1.01*self.n_eq,20.0),x0=100.0*self.n_eq)
     except ValueError:
       print("ValueError in rho_central_cyl")
       return (-1,-1)
     rho0 = res.root
     mu=self.gasprof.mu()
     rgas = self.get_rgas(rho0/mu)
     return res.root, rgas

  def rho_central_cyl(self,rho0trial,Rsfr,sfr):
     facHe = 1.36 #Hydrogen mass to total mass conversion factor
     mu=self.gasprof.mu()
     self.rgas=self.get_rgas(rho0trial/mu)
     print(self.rgas)
     if (self.rgas == -1):
        return 100#-1
     rhoH = dblquad(self.sfrHdens_cyl, 0, Rsfr, lambda x: 0, self.ub)
     rho0 = facHe*(sfr/rhoH[0])**(1.0/self.sfr_exp)
     return rho0-rho0trial

  def ub(self,x):
    if (x > self.rgas): return 0.0
    return np.sqrt(self.rgas**2-x**2)

  def get_rgas(self,n0trial):
     P_hot=self.n_hot*self.T_hot
     try:
       res=root_scalar(lambda r: n0trial*self.gasprof.ndens(r)*self.gasprof.T - P_hot,bracket=(0.001,10.0),x0=0.1) 
     except ValueError:
       print("ValueError in get_rgas")
       return -1
     return res.root

  def sfrHdens_cyl(self,z,R): 
     rho_convfac = 0.02439303604 #conversion factor Msun/pc**3 -> amu/cm**3
     r = np.sqrt(R**2+z**2)
     return 4.0*np.pi*self.sfr_fac*rho_convfac**self.sfr_exp*np.exp(-self.sfr_exp*self.gasprof.potentialdiff(r)/self.gasprof.cs())*R

  def mass(self, n0):
     rgas = self.get_rgas(n0)
     if rgas != -1:
        mass=self.gasprof.menc(rgas)
     else:
        mass = -1
     return (rgas,mass)

