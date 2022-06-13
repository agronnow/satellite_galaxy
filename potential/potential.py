import numpy as np
from scipy.integrate import quad
from abc import ABC, abstractmethod

class potential(ABC):
  @abstractmethod
  def potential(self, r):
     pass

  @abstractmethod
  def pot_zero(self):
     pass

  def potentialdiff(self, r):
     #calculate Phi(r) - Phi(0)
     return self.potential(r) - self.pot_zero()
