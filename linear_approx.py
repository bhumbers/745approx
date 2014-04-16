#!/usr/bin/env python

import os

import numpy as np
from sklearn import linear_model

from approx_generator import ApproxGenerator
import ioutils

class LinearApproxGenerator(ApproxGenerator):
  """
  Creates function approximator where entries of output grid are basic linear combinations of inputs
  Specifically, for output entry row r, column c: out_(r,c) = [in_p^T, 1] dot weights, where weights is (p+1) x 1
  This means that we train n_r * n_c * (p+1) weights ((p+1) of them per grid cell)
  (Note that the final "+1" is for constant intercept offset)
  """

  def __init__(self, params):
    self.weights = []

  def typename(sef):
    return "linear"

  def train(self, inputs, outputs):
    p = inputs.shape[1]       #number of input features
    n_r = outputs.shape[1]    #size of output grid in rows
    n_c = outputs.shape[2]    #size of output grid in cols

    assert inputs.shape[0] == outputs.shape[0]

    self.weights = np.zeros([n_r,n_c,p+1])

    #Train the linear weights for each output entry
    for r in xrange(n_r):
      for c in xrange(n_c):
        y = outputs[:,r,c] #output values for this cell over all observations
        X = inputs
        regr = linear_model.LinearRegression(fit_intercept=True, normalize=False)
        regr.fit(X, y)
        self.weights[r,c,:p] = regr.coef_
        self.weights[r,c,p] = regr.intercept_

    # print ('Train weights: \n', self.weights)

    # #Step 1: Find the min/max over each input dimension
    # minPerDim = inputs.min(axis=1)
    # maxPerDim = inputs.max(axis=1)
    # rangeSizePerDim = maxPerDim - minPerDim
    # rangeStepPerDim = rangeSizePerDim / self.tableSizePerDim

  def generate(self, out_path, out_file, out_func_name):
    ioutils.mkdir_p(out_path)

    n_r, n_c, p_plus_1 = self.weights.shape

    with open(os.path.join(out_path, out_file), 'w') as f:
      print >>f, '#include <stdio.h>'

      #Write regression weights W as a static const array
      print >>f, 'static const double W[%d][%d][%d] = {' % self.weights.shape
      print >>f, ',\n\n'.join(',\n'.join(', '.join('%f' % x for x in a) for a in b) for b in self.weights)
      print >>f, '};'

      #Write misc. constants such as in & output size
      print >>f, 'static const int p = %d, n_r = %d, n_c = %d;' % (p_plus_1 - 1, n_r, n_c)

      #Standard function signature for our compiler
      #(TODO: move this to shared func?)
      print >>f, 'void %s(double input[p], int inputLen, double output[n_r][n_c], int outputRows, int outputCols){' % out_func_name

      #Write nested loop summation code for the function call
      print >>f, '''
  for (int r = 0; r < n_r; r++) {
    for (int c = 0; c < n_c; c++) {
      double val = W[r][c][p];
      for (int i = 0; i < p; i++) {
        val += input[i] * W[r][c][i];
      }
      output[r][c] = val;
    }
  }
}
'''
