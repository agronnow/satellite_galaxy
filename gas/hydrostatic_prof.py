import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from ..potential import potential

class hydrostatic_prof:
  def __init__(self, pots, n0, T, r0=None):
    self.kB = 1.3806e-16/(1.67e-24*1.03e6**2) #Boltzman constant in RAMSES units
    self.pots = pots
    self.T = T
    self.T_tab, self.mu_tab=np.loadtxt('/Users/users/gronnow/src/Tmu_ramses.txt',unpack=True,skiprows=1)
    if r0 is None:
        self.n0 = n0
    else:
        res=root_scalar(self.nroot,args=(r0,n0),bracket=(0.0001,1.0),x0=0.002) 
        self.n0=res.root

  def nroot(self, n0, pt_r, pt_n):
    self.n0 = n0
    return self.ndens(pt_r) - pt_n

  def potentialdiff(self, r):
    potR = 0.0
    pot0 = 0.0
    for pot in self.pots:
      potR += pot.potential(r)
      pot0 += pot.pot_zero()
    return potR-pot0

  def ndens(self, r):
    return self.n0*np.exp(-self.potentialdiff(r)/self.cs())

  def mu(self):
    mu = np.interp(self.T,self.T_tab,self.mu_tab)
    return mu

  def cs(self):
     return self.kB*self.T/self.mu()

  def dmenc(self,r): 
     return 4*np.pi*r**2*self.ndens(r)

  def menc(self,r):
     mass=quad(self.dmenc,0,r)
     return self.mu()*mass[0]
