#!/usr/bin/env python

from approx_generator import ApproxGenerator
import ioutils
import numpy as np
from sklearn import linear_model

class LinearApproxGenerator(ApproxGenerator):
  """
  Creates function approximator where entries of output grid are basic linear combinations of inputs
  Specifically, for output entry row r, column c: out_(r,c) = [in_p^T, 1] dot weights, where weights is (p+1) x 1
  This means that we train n_r * n_c * (p+1) weights ((p+1) of them per grid cell)
  (Note that the final "+1" is for constant intercept offset)
  """

  def __init__(self, params):
    self.weights = []
    pass

  def typename(sef):
    return "linear"

  def train(self, inputs, outputs):
    n_r = outputs.shape[0]    #size of output grid in rows
    n_c = outputs.shape[1]    #size of output grid in cols
    p = inputs.shape[1]       #number of input features
    n = inputs.shape[0]       #number of observations (this should be the same as outputs.shape[2])

    #Train the linear weights for each output entry
    for r in xrange(n_r):
      for c in xrange(n_c):
        y = outputs[r,c,:] #output values for this cell over all observations
        X = inputs
        regr = linear_model.LinearRegression()
        regr.fit(X, y)
        w = regr.coef_
        print ('Train weights: \n', w)


    # #Step 1: Find the min/max over each input dimension
    # minPerDim = inputs.min(axis=1)
    # maxPerDim = inputs.max(axis=1)
    # rangeSizePerDim = maxPerDim - minPerDim
    # rangeStepPerDim = rangeSizePerDim / self.tableSizePerDim

    pass

  def generate(self, out_path, out_file, out_func_name):
    ioutils.mkdir_p(out_path)

    with open(out_path+out_file, 'w') as f:
      #Standard function signature for our compiler
      #(TODO: move this to shared func?)
      f.write('void ')
      f.write(out_func_name)
      f.write('(double* input, int inputLen, double* output, int outputRows, int outputCols)\n')
      f.write('{\n')
      f.write('}')


    #TODO: Write the basic function sig (need func name...)

    #TODO: Write regression weights (as a static const array)

    #TODO: Write nested loop summation code for the function call
