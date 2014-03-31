#!/usr/bin/env python

import abc

class ApproxGenerator:
  __metaclass__ = abc.ABCMeta
  """
  Generates C functions that approximate a given original function
  """

  @abc.abstractmethod
  def typename(sef):
    """
    Returns a short name/acronym that identifies this generator type
    """
    pass

  @abc.abstractmethod
  def train(self, inputs, outputsList):
    """Trains this generator to approximate function for given test cases
    inputs should be a 2D array where each row is an individual test input
    outputsList should be a list of the expected 2D grid output,
    where each entry is the 2D grid output for the corresponding input row
    """

    #NOTE: Feel free to do just about anything here, like gen & compling temp
    # C code in order to run test

    #TODO: Do we need to consider the effects of different grid sizes at this point,
    # or just assume that all grids are same size or else it doesn't matter? -BH
    pass

  @abc.abstractmethod
  def generate(self, path = '.'):
    """
    Writes a C file which represents this trained approximation to current
    directory (or path if given). Output file will be $(name).c
    """
    pass
