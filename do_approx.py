#!/usr/bin/env python

from ctypes import *
import numpy as np
import func_compiler
import math

import random

import os
import shutil

from sklearn.metrics import mean_squared_error
from math import sqrt

from collections import namedtuple

import timeit

import itertools

ApproxConfig = namedtuple('ApproxConfig', ['module', 'gen_class', 'name', 'params'])

#Sets inputs, outputs, etc. for bound ctype function for our approximator signature
def set_signature(func_handle):
  func_handle.restype = None
  func_handle.argtypes = [np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
                   c_int,
                   np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
                   c_int,
                   c_int]

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

def generate_mdp_inputs(num_inputs, num_rewards):
  rnd = random.Random()
  rnd.seed(42)
  input_length =   1 + 3*num_rewards # length of each input array
  func_inputs = np.zeros([num_inputs, input_length])
  for i in xrange(num_inputs):
    func_inputs[i, 0] = 0.99 #discount factor for MDP
    #Lay out some rewards
    for j in xrange(num_rewards):
      offset = 1 + 3*j
      func_inputs[i, offset+0] = rnd.random()   #position
      func_inputs[i, offset+1] = rnd.random()
      func_inputs[i, offset+2] = rnd.random()  #reward
  return func_inputs

def compute_func_results(func_source, func_name, func_inputs, out_rows, out_cols, results_dir):
  #Define the C function interface nicely
  #Source: http://stackoverflow.com/questions/5862915/passing-np-arrays-to-a-c-function-for-input-and-output
  orig_func = func_compiler.compile(func_source, func_name)
  set_signature(orig_func)

  save_results = True
  func_output = np.zeros([out_rows, out_cols])
  if (save_results):
    shutil.rmtree(results_dir) #clear any existing results data
    os.makedirs(results_dir)
  for inputIdx, inputRow in enumerate(func_inputs):
    func_output[:] = 0   #reset output before executing this call
    orig_func(inputRow, inputRow.size, func_output, out_rows, out_cols)
    # print(('Input #%d: \n' + str(inputRow) + '\n') % testIdx)
    # print('Output: \n' + str(funcOutput) + '\n')
    if save_results:
      np.save(results_dir + '/' + func_name + '_%04d.in' % inputIdx, inputRow)
      np.save(results_dir + '/' + func_name + '_%04d.out' % inputIdx, func_output)
  # return func_output

def load_func_in_out_data(results_dir):
  #First, load all the training in/out data for the function
  import glob
  inResFiles = sorted(glob.glob(results_dir + '/*.in*'))
  outResFiles = sorted(glob.glob(results_dir + '/*.out*'))
  assert(len(inResFiles) == len(outResFiles))

  num_entries = len(inResFiles)

  inArray = None
  outArray = None
  for entryIdx, (inResFile, outResFile) in enumerate(itertools.izip(inResFiles, outResFiles)):
    inVals  = np.load(inResFile)
    outVals = np.load(outResFile)

    #Since we require all inputs to be a single length (likewise for outputs),
    #we can figure the size we need to preallocate based on the first entry
    if inArray is None:
      inArray = np.zeros((num_entries, inVals.shape[0]))
      outArray = np.zeros((num_entries, outVals.shape[0], outVals.shape[1]))

    inArray[entryIdx,:] = inVals
    outArray[entryIdx,:,:] = outVals
  return inArray, outArray

#Train & write C function approximators of named function & return a list of generated approximator source files
def generate_approximators(approx_configs, func_name, trainIn, trainOut, approx_out_dir):
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
  return approx_files

def get_approx_errors(approx_files, func_name, testIn, testOut):
  #For each approximator, evaluate approximator quality on test set, save or show to console
  approx_errors = []
  approx_outputs_list = []
  for approx_file in approx_files:
    approx_outputs = np.zeros(testOut.shape)
    approx_func = func_compiler.compile(approx_file, func_name)
    set_signature(approx_func)

    approx_errors_by_test = np.zeros(len(testIn))
    #Find error of each approximation output compared to the original output
    for inputIdx, inputRow in enumerate(testIn):
      (out_rows, out_cols) = testOut[0,:,:].shape
      approx_func(inputRow, inputRow.size, approx_outputs[inputIdx,:,:], out_rows, out_cols)
      error = sqrt(mean_squared_error(testOut[inputIdx,:,:], approx_outputs[inputIdx,:,:])) #RMSE. We probably want to normalize this as well...
      approx_errors_by_test[inputIdx] = error
    approx_errors.append(np.average(approx_errors_by_test))
    approx_outputs_list.append(approx_outputs)

  print('RMSE BY APPROXIMATOR:')
  for approxIdx, approx_error in enumerate(approx_errors):
    approx_name = approx_files[approxIdx]
    print('  %s: %f' % (approx_name, approx_error))
  return approx_errors, approx_outputs_list

def get_approx_timing(approx_files, func_name, func_inputs, out_rows, out_cols):
  approx_timings = []
  for approx_file in approx_files:
    approx_timings.append(profile_function_speed(approx_file, func_name, func_inputs, out_rows, out_cols))

  print('TIME TO EXECUTE %d FUNCTION CALLS BY APPROXIMATOR:' % func_inputs.shape[0])
  for approxIdx, approx_timing in enumerate(approx_timings):
    approx_name = approx_files[approxIdx]
    print('  %s: %f' % (approx_name, approx_timing))
  return approx_timings

  # print('Profiled time to run %d inputs for source %s: %f' % (func_inputs.shape[0], func_source, runtime))

def profile_function_speed(func_source, func_name, func_inputs, out_rows, out_cols):
  func = func_compiler.compile(func_source, func_name)
  set_signature(func)

  def run_on_inputs(func, func_inputs, out_rows, out_cols):
    func_output = np.zeros([out_rows, out_cols])
    for inputIdx, inputRow in enumerate(func_inputs):
      # func_output[:] = 0   #reset output before executing this call
      func(inputRow, inputRow.size, func_output, out_rows, out_cols)

  t = timeit.Timer(lambda: run_on_inputs(func, func_inputs, out_rows, out_cols))
  runtime = t.timeit(number=1)
  return runtime


if __name__ == '__main__':
  #Script to execute approximation function generation for a given C/C++ function (in a shared lib)

  #TODO: Read JSON config with
  # - Input function(s) to approximate
  #   - Output file names (if provided, else use defaults)
  #   - Input generators of some sort...
  # - Approximation config parameters (which to try, what params to use for each)


  results_dir = './results'

  # Option #1: Sum-of-Gaussians
  func_name = 'sum_of_gaussians'
  func_source = './inputs/gaussian.c'
  input_gen = lambda: generate_sum_of_gaussians_inputs(num_inputs, 3)

  # # Option #2: MDP
  # func_name = 'compute_mdp_values'  #"basic_example"
  # func_source = './inputs/mdp.c' #"./inputs/basic.c"
  # input_gen = lambda: generate_mdp_inputs(num_inputs, 1)

  #Set up approximator configs of interest
  approx_configs = [
      ApproxConfig('dummy_approx', 'DummyApproxGenerator', 'dummy', {'src': func_source}),
      ApproxConfig('linear_approx', 'LinearApproxGenerator', 'linear', {})
    ]

  #Generate some random sum-of-Gaussians inputs
  print('Input generation...')
  num_inputs = 10000
  func_inputs = input_gen()

  #ORIGNAL FUNCTION: Compute grid output for each input row
  print('Computing original function results...')
  out_rows = 100            # Size in rows of grid output
  out_cols = 100             # Size in cols of grid output
  compute_func_results(func_source, func_name, func_inputs, out_rows, out_cols, results_dir)

  #APPROXIMATORS: Train & generate C function to replace original function
  # (NOTE: This part can & should be split out from in/out pair generation for original function)

  print('Generator training...')
  #Split data into training & test sets
  trainingPercent = 0.5
  inArray, outArray = load_func_in_out_data(results_dir)
  N = inArray.shape[0]
  Ntrain = math.floor(N * trainingPercent)
  Ntest = N - Ntrain
  trainIn = inArray[0:Ntrain,:]
  trainOut = outArray[0:Ntrain,:,:]
  testIn = inArray[Ntrain:,:]
  testOut = outArray[Ntrain:,:,:]

  #Then train & generate approximator outputs for different generators
  approx_out_dir = './approx/'
  approx_files = generate_approximators(approx_configs, func_name, trainIn, trainOut, approx_out_dir)

  #Approximator evaluation
  print('Errors eval...')
  approx_errors, approx_outputs_list = get_approx_errors(approx_files, func_name, testIn, testOut)
  print('Performance eval...')
  get_approx_timing(approx_files, func_name, testIn, out_rows, out_cols)
  print('DONE')
