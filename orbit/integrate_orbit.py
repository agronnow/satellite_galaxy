import numpy as np
from satellite_init.potential import potential

class integrate_orbit:
   def __init__(self,pot,evol=False,res=1):
      self.pot = pot
      self.evol = evol
      self.res = res

   def integrate(self,r0,v0,tend,dt,t0=0.0,method='leapfrog',backwards=True):
      ndim = len(r0)
      t0 *= 10.5347
      tend *= 10.5347
      niter = abs(int((tend-t0)/dt)-1)
      if (ndim==3):
         vn = np.array([v0[0]/10.3076,v0[1]/10.3076,v0[2]/10.3076])
      else:
         vn = np.array([v0[0]/10.3076,v0[1]/10.3076])
      rn = np.array(r0)
      an = self.acceleration(r0,t0)
      r = np.zeros(shape=(niter+1,ndim))
      v = np.zeros(shape=(niter+1,ndim))
      a = np.zeros(shape=(niter+1,ndim))
      r[0,:] = rn
      v[0,:] = vn
      a[0,:] = an
      times = np.zeros(niter+1)
      times[0] = t0
      i=0
      t=t0
      if backwards:
         dt = -dt
      for i in range(1,niter+1):
         rn, vn, an = self.step(rn, vn, an, dt, t, method)
         t = t + dt
         times[i] = t
         r[i,:] = rn
         v[i,:] = vn
         a[i,:] = an
      return times[::self.res]/10.5347,r[::self.res],v[::self.res]*10.3076,a[::self.res] #Time in Gyr, coordinates in kpc, velocities in km/s, acceleration in code units

   def step(self, rn, vn, an, dt, t, method):
      if (method=='euler'):
         rnp1 = rn + dt*vn
         anp1 = self.acceleration(rnp1)
         vnp1 = vn + dt*anp1
      else:
         rnp1 = rn + vn*dt + 0.5*an*dt*dt
         anp1 = self.acceleration(rnp1,t)
         vnp1 = vn + 0.5*dt*(an+anp1)
      return rnp1, vnp1, anp1

   def acceleration(self, pos, t):
      if len(pos) == 3:
         r = np.sqrt(pos[0]**2+pos[1]**2+pos[2]**2)
         z = pos[2]
      elif len(pos) == 2:
         r = np.sqrt(pos[0]**2+pos[1]**2)
         z = 0
      else:
         raise ValueError('Position vector must have 2 or 3 elements')

      accel_R = 0
      accel_z = 0
      if self.evol:
         accel = self.pot.acceleration(r, 1.e9*t/10.5347)
      else:
         accel = self.pot.acceleration(r)

      if type(accel) is not tuple:
         accel_r = accel
      elif len(accel)==2:
         accel_r, accel_R = accel
      elif len(accel)==3:
         accel_r, accel_R, accel_z = accel
      else:
         raise ValueError('Potential returned acceleration vector with more than 3 elements')

      if len(pos) == 3:
         return np.array([pos[0]*(accel_R+accel_r)/r, pos[1]*(accel_R+accel_r)/r, pos[2]*(accel_z+accel_r)/r])
      else:
         return np.array([pos[0]*accel_r/r, pos[1]*accel_r/r])

