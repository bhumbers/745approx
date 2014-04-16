#!/usr/bin/env python

from ctypes import *
import numpy as np
from numpy import linalg as LA
import ioutils
import func_compiler
import math

import random

import os
import shutil

from sklearn.metrics import mean_squared_error
from math import sqrt

from collections import namedtuple

# from dummy_approx import DummyApproxGenerator
# from lookup_approx import LookupApproxGenerator

#Sets inputs, outputs, etc. for bound ctype function for our approximator signature
def set_signature(func_handle):
  func_handle.restype = None
  func_handle.argtypes = [np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
                   c_int,
                   np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
                   c_int,
                   c_int]

ApproxConfig = namedtuple('ApproxConfig', ['module', 'gen_class', 'name', 'params'])

def generate_sum_of_gaussians_inputs(num_inputs, num_gaussians):
  rnd = random.Random()
  rnd.seed(42)
  input_length =   5*num_gaussians # length of each input array
  func_inputs = np.zeros([num_inputs, input_length])
  for i in xrange(num_inputs):
    #Lay out some Gaussians (normalized [0,1] range)
    for j in xrange(num_gaussians):
      offset = 5*j
      func_inputs[i, offset+0] = rnd.random()  #mean
      func_inputs[i, offset+1] = rnd.random()
      func_inputs[i, offset+2] = 0.2             #variance
      func_inputs[i, offset+3] = 0.4
      func_inputs[i, offset+4] = 3              #amplitude
  return func_inputs

def compute_func_results(func_inputs, out_rows, out_cols, results_dir):
  save_results = True
  num_inputs = func_inputs.shape[0]
  func_output = np.zeros([out_rows, out_cols])
  if (save_results):
    shutil.rmtree(results_dir)
    os.makedirs(results_dir)
  for inputIdx, inputRow in enumerate(func_inputs):
    func_output[:] = 0   #reset output before executing this call
    orig_func(inputRow, inputRow.size, func_output, out_rows, out_cols)
    # print(('Input #%d: \n' + str(inputRow) + '\n') % testIdx)
    # print('Output: \n' + str(funcOutput) + '\n')
    if save_results:
      np.save(results_dir + '/' + func_name + '_%04d.in' % inputIdx, inputRow)
      np.save(results_dir + '/' + func_name + '_%04d.out' % inputIdx, func_output)
  return func_output

def load_func_in_out_data(results_dir):
  #First, load all the training in/out data for the function
  import glob
  inResFiles = sorted(glob.glob(results_dir + '/*.in*'))
  outResFiles = sorted(glob.glob(results_dir + '/*.out*'))
  assert(len(inResFiles) == len(outResFiles))
  inList = []
  outList = []
  for inResFile in inResFiles:
    inList.append(np.load(inResFile))
  for outResFile in outResFiles:
    outList.append(np.load(outResFile))
  inArray = np.vstack(inList) if len(inList) > 0 else []
  outArray = np.dstack(outList) if len(outList) > 0 else []
  return inArray, outArray

def get_approx_errors(approx_files, func_name, testIn, testOut):
  #For each approximator, evaluate approximator quality on test set, save or show to console
  approx_errors = []
  for approx_file in approx_files:
    approx_func = func_compiler.compile(approx_file, func_name)
    set_signature(approx_func)

    approx_errors_by_test = np.zeros(len(testIn))
    #Find error of each approximation output compared to the original output
    for inputIdx, inputRow in enumerate(testIn):
      orig_output = testOut[:,:,inputIdx]
      (out_rows, out_cols) = orig_output.shape
      approx_output = np.zeros([out_rows, out_cols])
      approx_func(inputRow, inputRow.size, approx_output, out_rows, out_cols)
      error = sqrt(mean_squared_error(orig_output, approx_output)) #RMSE. We probably want to normalize this as well...
      approx_errors_by_test[inputIdx] = error
    approx_errors.append(np.average(approx_errors_by_test))

  print('RMSE BY APPROXIMATOR:')
  for approxIdx, approx_error in enumerate(approx_errors):
    approx_name = approx_configs[approxIdx][1]
    print('  %s: %f' % (approx_name, approx_error))
  return approx_errors


if __name__ == '__main__':
  #Script to execute approximation function generation for a given C/C++ function (in a shared lib)

  #TODO: Read JSON config with
  # - Input function(s) to approximate
  #   - Output file names (if provided, else use defaults)
  #   - Input generators of some sort...
  # - Approximation config parameters (which to try, what params to use for each)
  func_name = 'sum_of_gaussians'  #"basic_example"
  func_source = './inputs/gaussian.c' #"./inputs/basic.c"
  results_dir = './results'

  #Set up approximator configs of interest
  #TODO: Shift these out to JSON config files, if convenient
  approx_configs = [
      ApproxConfig('dummy_approx', 'DummyApproxGenerator', 'dummy', {'src': func_source}),
      ApproxConfig('linear_approx', 'LinearApproxGenerator', 'lookup', {'tableSizePerDim': 100})
    ]

  #Generate some random sum-of-Gaussians inputs
  num_inputs = 1000
  num_gaussians = 3
  func_inputs = generate_sum_of_gaussians_inputs(num_inputs, num_gaussians)

  #Define the C function interface nicely
  #Source: http://stackoverflow.com/questions/5862915/passing-np-arrays-to-a-c-function-for-input-and-output
  orig_func = func_compiler.compile(func_source, func_name)
  set_signature(orig_func)

  #ORIGNAL FUNCTION: Compute grid output for each input row
  out_rows = 10             # Size in rows of grid output
  out_cols = 10             # Size in cols of grid output
  compute_func_results(func_inputs, out_rows, out_cols, results_dir)

  #APPROXIMATORS: Train & generate C function to replace original function
  # (NOTE: This part can & should be split out from in/out pair generation for original function)

  #Split data into training & test sets
  inArray, outArray = load_func_in_out_data(results_dir)
  N = inArray.shape[0]
  trainingPercent = 0.5
  Ntrain = math.floor(N * trainingPercent)
  Ntest = N - Ntrain
  trainIn = inArray[0:Ntrain,:]
  trainOut = outArray[:,:,0:Ntrain]
  testIn = inArray[Ntrain:,:]
  testOut = outArray[:,:,Ntrain:]

  #Then train & generate approximator outputs for different generators

  approx_out_dir = './approx/'
  approx_files = []
  for approx_config in approx_configs:
    approx_name = approx_config.name
    approx_out_file = approx_name + '.c'
    approx_module = __import__(approx_config.module)
    approx_gen_class = getattr(approx_module, approx_config.gen_class)
    approx_gen = approx_gen_class(approx_config.params)
    approx_gen.train(trainIn, trainOut)
    approx_gen.generate(approx_out_dir, approx_out_file, func_name)
    approx_files.append(approx_out_dir + approx_out_file)

  get_approx_errors(approx_files, func_name, testIn, testOut)
