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

#Script to execute approximation function generation for a given C/C++ function (in a shared lib)

rnd = random.Random()
rnd.seed(42)

saveFuncResults = True
resultsDir = './results'

#TODO: Read JSON config with
# - Input function(s) to approximate
#   - Output file names (if provided, else use defaults)
#   - Input generators of some sort...
# - Approximation config parameters (which to try, what params to use for each)
#PLACEHOLDER: Basic lib loading & execution (run "./inputs/make" before this)
inputFunc = 'sum_of_gaussians'  #"basic_example"
inputFuncSource = './inputs/gaussian.c' #"./inputs/basic.c"

#Set up approximator configs of interest
#TODO: Shift these out to JSON config files, if convenient
approx_configs = [
    ApproxConfig('dummy_approx', 'DummyApproxGenerator', 'dummy', {'src': inputFuncSource}),
    ApproxConfig('linear_approx', 'LinearApproxGenerator', 'lookup', {'tableSizePerDim': 100})
  ]


#PLACEHOLDER: Generate some arbitrary inputs (one row per call)
numInputTests = 2     # number of distinct input instances to test
inputLen = 5            # length of each input array
numOutRows = 10         # Size in rows of grid output
numOutCols = 10         # Size in cols of grid output
funcInputs = np.zeros([numInputTests, inputLen])
for i in range(numInputTests):
  #Simple Gaussian centered in middle of grid (normalized [0,1] range)
  funcInputs[i, 0] = rnd.random()  #mean
  funcInputs[i, 1] = rnd.random()
  funcInputs[i, 2] = 0.2  #variance
  funcInputs[i, 3] = 0.4
  funcInputs[i, 4] = 3    #amplitude
funcOutput = np.zeros([numOutRows, numOutCols])

if (saveFuncResults):
  ioutils.mkdir_p(resultsDir)

#Define the C function interface nicely
#Source: http://stackoverflow.com/questions/5862915/passing-np-arrays-to-a-c-function-for-input-and-output
orig_func = func_compiler.compile(inputFuncSource, inputFunc)
set_signature(orig_func)

#ORIGNAL FUNCTION: Compute grid output for each input row
shutil.rmtree(resultsDir)
os.makedirs(resultsDir)
for testIdx, inputRow in enumerate(funcInputs):
  funcOutput[:] = 0   #reset output before executing this call
  orig_func(inputRow, inputRow.size, funcOutput, numOutRows, numOutCols)
  # print(('Input #%d: \n' + str(inputRow) + '\n') % testIdx)
  # print('Output: \n' + str(funcOutput) + '\n')
  if saveFuncResults:
    np.save(resultsDir + '/' + inputFunc + '_%04d.in' % testIdx, inputRow)
    np.save(resultsDir + '/' + inputFunc + '_%04d.out' % testIdx, funcOutput)

#APPROXIMATORS: Train & generate C function to replace original function
# (NOTE: This part can & should be split out from in/out pair generation for original function)

#First, load all the training in/out data for the function
import glob
inResFiles = sorted(glob.glob(resultsDir + '/*.in*'))
outResFiles = sorted(glob.glob(resultsDir + '/*.out*'))
assert(len(inResFiles) == len(outResFiles))
inList = []
outList = []
for inResFile in inResFiles:
  inList.append(np.load(inResFile))
for outResFile in outResFiles:
  outList.append(np.load(outResFile))
inArray = np.vstack(inList) if len(inList) > 0 else []
outArray3d = np.dstack(outList) if len(outList) > 0 else []

#Split data into training & test sets
#PLACEHOLDER: Basic split
N = inArray.shape[0]
trainingPercent = 0.5
Ntrain = math.floor(N * trainingPercent)
Ntest = N - Ntrain
trainIn = inArray[0:Ntrain,:]
trainOut = outArray3d[:,:,0:Ntrain]
testIn = inArray[Ntrain:,:]
testOut = outArray3d[:,:,Ntrain:]

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
  approx_gen.generate(approx_out_dir, approx_out_file, inputFunc)
  approx_files.append(approx_out_dir + approx_out_file)

#For each approximator, evaluate approximator quality on test set, save or show to console
approx_errors = []
for approx_file in approx_files:
  approx_func = func_compiler.compile(approx_file, inputFunc)
  set_signature(approx_func)

  approx_errors_by_test = np.zeros(len(testIn))
  #Find error of each approximation output compared to the original output
  for inputIdx, inputRow in enumerate(testIn):
    orig_output = testOut[:,:,inputIdx]
    (out_rows, out_cols) = orig_output.shape
    approx_output = np.zeros([out_rows, out_cols])
    approx_func(inputRow, inputRow.size, approx_output, out_rows, out_cols)
    error = LA.norm((approx_output - orig_output))
    approx_errors_by_test[inputIdx] = error
  approx_errors.append(np.average(approx_errors_by_test))

print('AVERAGE APPROX ERRORS:')
for approxIdx, approx_error in enumerate(approx_errors):
  approx_name = approx_configs[approxIdx][1]
  print('  %s: %f' % (approx_name, approx_error))
