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
    #n = inputs.shape[0]       #number of observations (this should be the same as outputs.shape[2])

    self.weights = np.zeros([n_r,n_c,p+1])

    #Train the linear weights for each output entry
    for r in xrange(n_r):
      for c in xrange(n_c):
        y = outputs[r,c,:] #output values for this cell over all observations
        X = inputs
        regr = linear_model.LinearRegression(fit_intercept=True, normalize=False)
        regr.fit(X, y)
        self.weights[r,c,:p] = regr.coef_
        self.weights[r,c,p] = regr.intercept_

    print ('Train weights: \n', self.weights)

    # #Step 1: Find the min/max over each input dimension
    # minPerDim = inputs.min(axis=1)
    # maxPerDim = inputs.max(axis=1)
    # rangeSizePerDim = maxPerDim - minPerDim
    # rangeStepPerDim = rangeSizePerDim / self.tableSizePerDim

    pass

  def generate(self, out_path, out_file, out_func_name):
    ioutils.mkdir_p(out_path)

    with open(out_path+out_file, 'w') as f:
      #Write regression weights W as a static const array
      f.write('static const double W[] = {\n')
      (n_r, n_c, p_plus_1) = self.weights.shape
      for r in xrange(n_r):
        for c in xrange(n_c):
          for i in xrange(p_plus_1):
            f.write('%f, ' % self.weights[r,c,i])
          f.write('\n')
        f.write('\n')
      f.seek(-4, 1)  #erase the extraneous ', ' (plus extra new lines) from the last entry in the array
      f.write('\n};\n')

      #Write misc. constants such as in & output size
      f.write('\n');
      f.write('static const int pPlusOne = %d;\n' % p_plus_1)
      f.write('static const int n_r = %d;\n' % n_r)
      f.write('static const int n_c = %d;\n' % n_c)
      f.write('\n');

      #Standard function signature for our compiler
      #(TODO: move this to shared func?)
      f.write('void ')
      f.write(out_func_name)
      f.write('(double* input, int inputLen, double* output, int outputRows, int outputCols)\n')
      f.write('{\n')

      #Write nested loop summation code for the function call
      f.write('  for (int r = 0; r < n_r; r++) {\n')
      f.write('    for (int c = 0; c < n_c; c++) {\n')
      f.write('      double val = 0;\n')
      f.write('      for (int i = 0; i < pPlusOne; i++) {\n')
      f.write('       //Sort of a bad place for a conditional, but need to use const 1 for last term\n')
      f.write('        double inVal = (i < pPlusOne) ? input[i] : 1;\n\n')
      f.write('        val += inVal * W[r*(n_c*pPlusOne) + c*pPlusOne + i];\n')
      f.write('      }\n;')
      f.write('      output[r*n_c + c] = val;\n')
      f.write('    }\n')
      f.write('  }\n')

      f.write('}')
