import numpy as np
from scipy.optimize import root_scalar
import galpy.potential
from galpy.util import conversion
from ..potential.nfw import nfw

def calc_rtidal(mass,tarr,rarr,rhodm0_sat,rs_sat,evol=True):
    if evol:
        pot_mw = nfw(mvir=mass, Msun=True, interp=True)
        pot_sat = nfw(rhos=rhodm0_sat, rs=rs_sat)
    else:
        pot = galpy.potential.MWPotential2014
        pot[2]*=mass/0.8e12

    rt=[]
    for t, r in zip(tarr, rarr):
        if evol:
            try:
               res=root_scalar(rtidalroot,args=(r,t,pot_sat,pot_mw),bracket=(1.0,100),x0=10.0)
               rt=np.append(rt,res.root)
            except ValueError:
               rt=np.append(rt,-1)
        else:
            try:
               res=root_scalar(rtidalroot_galpy,args=(r,rhodm0_sat,rs_sat,pot),bracket=(1.0,100),x0=10.0)
               rt=np.append(rt,res.root)
            except ValueError:
               rt=np.append(rt,-1)
    return rt

def rtidalroot(rt, r, t, pot_sat, pot_mw):
    return rt-r*(pot_sat.mass_enclosed(rt)/(3.0*pot_mw.mass_enclosed(r,t)))**(1./3.)

def rtidalroot_galpy(rt,r,rhodm0_sat,rs_sat,pot):
    return rt-r*(mass(rt,rhodm0_sat,rs_sat)/(3.0*(pot[0].mass(r/8.0)+pot[1].mass(r/8.0)+pot[2].mass(r/8.0))*conversion.mass_in_msol(220.,8.)/24539982.))**(1./3.)

