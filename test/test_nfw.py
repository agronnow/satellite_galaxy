import pytest
import numpy as np
from ..potential.nfw import nfw

def test_nfw_potential():
    pot = nfw(rs=1.46, rhos=1.12)
    r_in = np.linspace(0.1, 10.1, 100)
    r_out = np.linspace(0.2, 10.2, 100)
    #Potential should always be negative
    assert np.all(pot.potential(r_in) < 0.0)
    #Potential should increase towards zero with increasing radius
    assert np.all(pot.potential(r_out) > pot.potential(r_in))

def test_nfw_calc_params():
    pot_rhos = nfw(rs=1.46, rhos=1.12)
    pot_mvir = nfw(mvir=1.e12, rvir=250.0, c=15.0, Msun=True)
    #After supplying virial radius this should be consistent with concentration and scale length
    pot_rhos.calc_params(rvir=1.0)
    assert pot_rhos.rvir == pytest.approx(pot_rhos.c * pot_rhos.rs)
    # Scale length calculated from virial radius should be consistent with concentration
    pot_mvir.calc_params()
    assert pot_mvir.rs == pytest.approx(pot_mvir.rvir / pot_mvir.c)

def test_nfw_evaluate_at_time():
    #Virial mass and radius should grow with time in evolving potential
    pot = nfw(mvir=1.e12, Msun=True)
    rvir_now = pot.rvir
    pot.evaluate_at_time(5.e9)
    assert pot.mvir < 1.e12
    assert pot.rvir < rvir_now

def test_nfw_acceleration():
    pot = nfw(rs=1.46, rhos=1.12)
    r_in = np.linspace(0.1, 10.1, 100)
    r_out = np.linspace(0.2, 10.2, 100)
    # Acceleration should point towards centre (i.e. be negative for r>0)
    assert np.all(pot.acceleration(r_in) < 0.0)
    # Acceleration should be increasingly negative towards centre
    assert np.all(pot.potential(r_out) > pot.potential(r_in))


def test_nfw_mass_enclosed():
    #Mass calculated within virial radius should equal initially supplied virial mass
    pot = nfw(mvir=1.e12, rvir=250.0, Msun=True)
    assert pot.mass_enclosed(pot.rvir) == pytest.approx(pot.mvir)
