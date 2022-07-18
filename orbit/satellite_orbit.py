import galpy
import galpy.potential
import galpy.orbit
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
from satellite_init.orbit.integrate_orbit import integrate_orbit
from satellite_init.potential.nfw import nfw
from numpy.random import default_rng
import multiprocessing
from itertools import repeat
from pandas import read_csv
import os


class satellite_orbit:

    def __init__(self, integrator=None, mass=0.0, evol=True, disk=False, use_dr2=False, lib="", res=100, mcdir=""):
        self.ts = []
        self.vels = []
        self.rads = []
        self.res = res
        self.mcdir = mcdir

        if lib == "galpy":
            nfwpot = galpy.potential.NFWPotential(a=2.0, amp=mass * 4.85223 / 0.8e12)  # mass in units of Msun
            if disk:
                self.pot = [
                    galpy.potential.PowerSphericalPotentialwCutoff(amp=1.25, r1=1.0 / 8.0, alpha=1.8, rc=1.9 / 8.0),
                    galpy.potential.MiyamotoNagaiPotential(a=3. / 8.0, b=0.28 / 8.0, amp=0.75748), nfwpot]
            else:
                self.pot = nfwpot
            self.integrator = None
        else:
            if integrator is None:
                if evol:
                    pot = nfw(mvir=mass, Msun=True)
                else:
                    pot = nfw(mvir=mass, rvir=245.0, c=15.3, Msun=True)  # Matches NFW potential in MW14 model
                self.integrator = integrate_orbit(pot, evol, res)
            else:
                self.integrator = integrator
            self.pot = None

        if use_dr2:
            self.galcen_dist = 8.2 * u.kpc
            self.galcen_vsun = coord.CartesianDifferential(11.0 * u.km / u.s, 248.0 * u.km / u.s, 7.3 * u.km / u.s)
            fname = "dsph_dr2.dat"
        else:
            # As used in Li+ 21 for GAIA EDR3
            self.galcen_dist = 8.122 * u.kpc
            self.galcen_vsun = coord.CartesianDifferential(10.0 * u.km / u.s, 240.0 * u.km / u.s, 7.0 * u.km / u.s)
            fname = "dsphs.dat"
        satellite_table = read_csv("dsphs.dat", sep="\s+", comment="#")
        self.orbit_data = satellite_table.set_index("Name").to_dict(orient="index")

    def calc_single_orbit(self, data, tbeg, tend, dm=None, pm_ra=None, pm_dec=None, rettime=False):
        # dm,ra,dec,data,tbeg,tend = args
        if dm is None:
            dist = data["dist"]
        else:
            dist = 10 ** (dm / 5.0 - 2.0)
        if pm_ra is None:
            pm_ra = data["pm_ra"]
        if pm_dec is None:
            pm_dec = data["pm_dec"]
        i = coord.SkyCoord(ra=data["ra"], dec=data["dec"], distance=dist * u.kpc,
                           pm_ra_cosdec=pm_ra * u.mas / u.yr, pm_dec=pm_dec * u.mas / u.yr,
                           radial_velocity=data["radvel"] * u.km / u.s, frame='icrs')
        g = i.transform_to(coord.Galactocentric(galcen_distance=self.galcen_dist, galcen_v_sun=self.galcen_vsun))
        if self.integrator is None:  # use galpy with given static pot
            ts = np.linspace(tbeg, tend, self.res) * u.Gyr
            o = galpy.orbit.Orbit(g)
            o.integrate(ts, self.pot)
            vel = np.zeros([len(ts), 2])
            r = np.zeros([len(ts), 2])
            vel[:, 0] = o.vx(ts)
            vel[:, 1] = o.vy(ts)
            vel[:, 2] = o.vz(ts)
            r[:, 0] = o.x(ts)
            r[:, 1] = o.y(ts)
            r[:, 2] = o.z(ts)
            ts = ts.value
        else:  # use custom integrator for evolving NFW pot
            p0 = [-g.x.value, g.y.value, g.z.value]
            v0 = [-g.v_x.value, g.v_y.value, g.v_z.value]
            ts, r, vel, f = self.integrator.integrate(p0, v0, tend, 0.001, tbeg)

        vel /= 10.3
        if rettime:
            return ts, r, vel
        else:
            return r, vel

    def calc_orbit(self, name, tbeg=0.0, tend=0.01, nmontecarlo=0, nproc=1, res=100, sample_only=False, from_table=False, fname = ""):
        try:
            data = self.orbit_data[name]
        except KeyError:
            print("No satellite galaxy named " + name + " in table dsphs.dat")
            return

        if self.integrator is None:
            niter = self.res
        else:
            if tbeg == 0.0:
                tbeg = self.integrator.pot.now() / 1.e9
            niter = abs(int((tend - tbeg) * 10.5347 / 0.001 / res))

        self.vels = np.zeros([niter + 1, 3, nmontecarlo + 1])
        self.rads = np.zeros([niter + 1, 3, nmontecarlo + 1])

        self.ts, r, vel = self.calc_single_orbit(data, tbeg, tend, rettime=True)
        self.vels[:, :, 0] = vel
        self.rads[:, :, 0] = r

        if nmontecarlo > 0:
            if from_table:
                pm_ra_cosdecs, pm_decs, dms = np.loadtxt("samples.dat",unpack=True)
            else:
                # randomly sample distances and proper motions based on observed uncertainties
                rng = default_rng()
                pm_ra_cosdecs = rng.normal(data["pm_ra"], data["pm_ra_err"], nmontecarlo)
                pm_decs = rng.normal(data["pm_dec"], data["pm_dec_err"], nmontecarlo)
                dms = rng.normal(data["distmod"], data["distmod_err"], nmontecarlo)

                np.savetxt("samples.dat",np.c_[pm_ra_cosdecs, pm_decs, dms])
                if sample_only:
                    return

            pool = multiprocessing.Pool(processes=nproc)
            orbits = pool.starmap(self.calc_single_orbit,
                                  zip(repeat(data), repeat(tbeg), repeat(tend), dms,
                                      pm_ra_cosdecs, pm_decs))
            # extract radius and velocity vectors from list of orbits
            self.rads = np.moveaxis(np.array(orbits)[:, 0, :, :], 0, 2)
            self.vels = np.moveaxis(np.array(orbits)[:, 1, :, :], 0, 2)

            if self.mcdir != "":
                np.savez_compressed(self.mcdir + "/orbits_" + name + fname + ".npz", ts=self.ts,
                                    rads=self.rads, vels=self.vels)

    def get_orbit(self, to1d=True):
        if to1d:
            rd = np.sqrt(self.rads[:, 0, :] ** 2 + self.rads[:, 1, :] ** 2 + self.rads[:, 2, :] ** 2)
            vd = np.sqrt(self.vels[:, 0, :] ** 2 + self.vels[:, 1, :] ** 2 + self.vels[:, 2, :] ** 2)
            return self.ts, rd, vd
        else:
            return self.ts, self.rads, self.vels
