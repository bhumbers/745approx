#!/usr/bin/env python

from approx_generator import ApproxGenerator
import shutil
import ioutils

class DummyApproxGenerator(ApproxGenerator):
  """
  A placeholder approximation generator that simply copies an input C file
  (eg: as a way of duplicating the original function implementation)
  """

  def __init__(self, orig_func_filepath):
    self.orig_func_filepath = orig_func_filepath

  def typename(self):
    return "dummy"

  def train(self, inputs, outputsList):
    pass

  def generate(self, out_path, out_file):
    #Dummy: Just write a duplicate of the original file
    ioutils.mkdir_p(out_path)
    shutil.copy2(self.orig_func_filepath, out_path+out_file)
