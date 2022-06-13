import galpy
import galpy.potential
import galpy.orbit
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
from .integrate_orbit import integrate_orbit
from ..potential.nfw import nfw
from numpy.random import default_rng
import multiprocessing
from itertools import repeat
import os

def get_orbit(pot,integrator,dm,ra,dec,data,n,mass,tbeg,tend):
    #dm,ra,dec,data,tbeg,tend = args
    i=coord.SkyCoord(ra=data[1][n],dec=data[2][n], distance=10**(dm/5.0-2.0)*u.kpc, pm_ra_cosdec=ra*u.mas/u.yr,pm_dec=dec*u.mas/u.yr,radial_velocity=data[9][n]*u.km/u.s,frame='icrs')
    g=i.transform_to(coord.Galactocentric(galcen_distance=8.122*u.kpc, galcen_v_sun=coord.CartesianDifferential(10.0*u.km/u.s, 240.0*u.km/u.s, 7.0*u.km/u.s)))
    if integrator is None: #use galpy with given static pot
        ts=np.linspace(tbeg,tend,10000)*u.Gyr
        o=galpy.orbit.Orbit(g)
        o.integrate(ts,pot)
        vel[:,0,0] = o.vx(ts)
        vel[:,1,0] = o.vy(ts)
        vel[:,2,0] = o.vz(ts)
        r[:,0,0] = o.x(ts)
        r[:,1,0] = o.y(ts)
        r[:,2,0] = o.z(ts)
    else: #use custom integrator for evolving NFW pot
        vR=g.v_x.value
        vT=np.sqrt(g.v_z.value**2+g.v_y.value**2)
        p0=[-g.x.value,g.y.value,g.z.value]
        v0=[-g.v_x.value,g.v_y.value,g.v_z.value]
        ts,r,vel,f=integrator.integrate(p0,v0,tend,0.001,tbeg)

    vel/=10.3
    return (r,vel)

def calc_orbit(name,mass,tbeg,tend,integrator,evol,disk,lib,datarel,nmontecarlo,nproc,res=100):
  use_dr2 = (datarel == "dr2")
  if integrator is not None:
      mass = integrator.pot.mvir
  if lib=="galpy":
      nfwpot = galpy.potential.NFWPotential(a=2.0,amp=mass*4.85223/0.8e12) #mass in units of Msun
      if (disk):
        pot = [galpy.potential.PowerSphericalPotentialwCutoff(amp=1.25,r1=1.0/8.0,alpha=1.8, rc=1.9/8.0),
               galpy.potential.MiyamotoNagaiPotential(a=3./8.0,b=0.28/8.0,amp=0.75748),nfwpot]
      else:
        pot = nfwpot


  ts=np.linspace(tbeg,tend,10000)*u.Gyr
  if use_dr2:
    fname = "dsph_dr2.dat"
  else:
    fname = "dsphs.dat"
  data=np.genfromtxt(fname,unpack=True,comments='#',dtype=None,encoding='utf8')
  vel=[]
  r=[]
  n=-1
  #fname,fra,fdec,fdist,fpm_ra,fpm_ra_err,fpm_dec,fpm_dec_err,fpm_syserr,fradvel
  for i in range(0,len(data[0])): 
    if data[0][i]==name: 
      n=i 
      break

  vels = []
  rads = []

  data[8][n]=0.0
  i=coord.SkyCoord(ra=data[1][n],dec=data[2][n], distance=data[3][n]*u.kpc, pm_ra_cosdec=data[4][n]*u.mas/u.yr,pm_dec=data[6][n]*u.mas/u.yr,radial_velocity=data[9][n]*u.km/u.s,frame='icrs')
  if use_dr2:
    g=i.transform_to(coord.Galactocentric(galcen_distance=8.2*u.kpc, galcen_v_sun=coord.CartesianDifferential(11.0*u.km/u.s, 248.0*u.km/u.s, 7.3*u.km/u.s)))
  else:
    g=i.transform_to(coord.Galactocentric(galcen_distance=8.122*u.kpc, galcen_v_sun=coord.CartesianDifferential(10.0*u.km/u.s, 240.0*u.km/u.s, 7.0*u.km/u.s))) #As used in Li+ 21 for GAIA EDR3
  if lib=="galpy":
    niter = 9999
    vels = np.zeros([niter+1,3,1])
    rads = np.zeros([niter+1,3,1])
    if nmontecarlo > 0:
       vels = np.zeros([niter+1,2,nmontecarlo+1])
       rads = np.zeros([niter+1,2,nmontecarlo+1])
    o=galpy.orbit.Orbit(g)
    o.integrate(ts,pot)
    vels[:,0,0] = o.vx(ts)/10.3
    vels[:,1,0] = o.vy(ts)/10.3
    vels[:,2,0] = o.vz(ts)/10.3
    rads[:,0,0] = o.x(ts)
    rads[:,1,0] = o.y(ts)
    rads[:,2,0] = o.z(ts)

    if nmontecarlo > 0:
        rng = default_rng()
        pm_ra_cosdecs = rng.normal(data[4][n], data[5][n], nmontecarlo)
        pm_decs = rng.normal(data[6][n], data[7][n], nmontecarlo)
        dms = rng.normal(data[12][n], data[13][n], nmontecarlo)
        pool = multiprocessing.Pool(processes=nproc)
        result = pool.starmap(get_orbit, zip(repeat(pot),repeat(None),dms,pm_ra_cosdecs,pm_decs,repeat(data),repeat(n),repeat(mass),repeat(tbeg),repeat(tend)))
        for imc in range(0,nmontecarlo):
            rads[:,0,1+imc] = result[imc][0]
            vels[:,0,1+imc] = result[imc][1]
        np.savez_compressed("/net/dataserver3/data/users/gronnow/orbits_"+name+"_edr3.npz",ts=ts.value,rads=rads,vels=vels)
  else:
    if integrator is None:
        if evol:
            pot = nfw(mvir=mass, Msun=True)
        else:
            pot = nfw(mvir=mass, rvir=245.0, c=15.3, Msun=True) #Matches NFW potential in MW14 model
        integrator = integrate_orbit(pot, evol, res)
    if tbeg < 0.0:
        tbeg = integrator.pot.now()/1.e9

    niter = abs(int((tend-tbeg)*10.5347/0.001/100))
    vels = np.zeros([niter+1,3,nmontecarlo+1])
    rads = np.zeros([niter+1,3,nmontecarlo+1])

    vR=g.v_x.value
    vT=np.sqrt(g.v_z.value**2+g.v_y.value**2)
    p0=[-g.x.value,g.y.value,g.z.value]
    v0=[-g.v_x.value,g.v_y.value,g.v_z.value]
    ts,r,vel,f=integrator.integrate(p0,v0,tend,0.001,tbeg)
    vel=vel/10.3
    vels[:,:,0] = vel
    rads[:,:,0] = r
    
    if nmontecarlo > 0:
        rng = default_rng()
        pm_ra_cosdecs = rng.normal(data[4][n], data[5][n], nmontecarlo)
        pm_decs = rng.normal(data[6][n], data[7][n], nmontecarlo)
        dms = rng.normal(data[12][n], data[13][n], nmontecarlo)
        pool = multiprocessing.Pool(processes=nproc)
        result = pool.starmap(get_orbit, zip(repeat(None),repeat(integrator),dms,pm_ra_cosdecs,pm_decs,repeat(data),repeat(n),repeat(mass),repeat(tbeg),repeat(tend)))
        for imc in range(0,nmontecarlo):
            rads[:,:,1+imc] = result[imc][0]
            vels[:,:,1+imc] = result[imc][1]

        np.savez_compressed("/net/dataserver3/data/users/gronnow/orbits_"+name+"_edr3.npz",ts=ts,rads=rads,vels=vels)

  return ts,rads,vels
