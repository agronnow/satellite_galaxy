import pytest
import numpy as np
from ..potential.nfw import nfw

def test_nfw_pot_negative():
    pot = nfw(rs=1.46, rhos=1.12)
    r = np.linspace(0.1, 10.0, 100)
    assert np.all(pot.potential(r) < 0.0)

def test_nfw_pot_increasing():
    pot = nfw(rs=1.46, rhos=1.12)
    r_in = np.linspace(0.1, 10.1, 100)
    r_out = np.linspace(0.2, 10.2, 100)
    assert np.all(pot.potential(r_out) > pot.potential(r_in))

def test_nfw_conversion_to_vir():
    pot = nfw(rs=1.46, rhos=1.12)
    pot.calc_params(rvir=1.0)
    assert pot.rvir == pytest.approx(pot.c * pot.rs)

def test_nfw_conversion_to_rho():
    pot = nfw(mvir=1.e12, rvir=250.0, Msun=True)
    pot.calc_params()
    assert pot.rs == pytest.approx(pot.rvir / pot.c)

def test_nfw_evolution():
    pot = nfw(mvir=1.e12, Msun=True)
    pot.evaluate_at_time(5.e9)
    assert pot.mvir < 1.e12

def test_nfw_mass_enclosed():
    pot = nfw(mvir=1.e12, rvir=250.0, Msun=True)
    assert pot.mass_enclosed(pot.rvir) == pytest.approx(pot.mvir)

