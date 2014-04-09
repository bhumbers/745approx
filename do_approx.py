#!/usr/bin/env python

from ctypes import *
import numpy as np
from numpy import linalg as LA
import ioutils
import func_compiler
import math

from dummy_approx import DummyApproxGenerator

#Sets inputs, outputs, etc. for bound ctype function for our approximator signature
def set_signature(func_handle):
  func_handle.restype = None
  func_handle.argtypes = [np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
                   c_int,
                   np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
                   c_int,
                   c_int]

#Script to execute approximation function generation for a given C/C++ function (in a shared lib)

saveFuncResults = True
resultsDir = './results'

#TODO: Read JSON config with
# - Input function(s) to approximate
#   - Output file names (if provided, else use defaults)
#   - Input generators of some sort...
# - Approximation config parameters (which to try, what params to use for each)
#PLACEHOLDER: Basic lib loading & execution (run "./inputs/make" before this)
inputFunc = "basic_example"
inputFuncSource = "./inputs/basic.c" #  only needed for DummyApproxGenerator

#PLACEHOLDER: Generate some arbitrary inputs (one row per call)
numInputTests = 100   # number of distinct input instances to test
inputLen = 5        # length of each input array
numOutRows = 10     # Size in rows of grid output
numOutCols = 10     # Size in cols of grid output
funcInputs = np.zeros([numInputTests, inputLen])
for i in range(numInputTests):
  for j in range(inputLen):
    funcInputs[i, j] = i
funcOutput = np.zeros([numOutRows, numOutCols])

if (saveFuncResults):
  ioutils.mkdir_p(resultsDir)

#Define the C function interface nicely
#Source: http://stackoverflow.com/questions/5862915/passing-np-arrays-to-a-c-function-for-input-and-output
orig_func = func_compiler.compile(inputFuncSource, inputFunc)
set_signature(orig_func)

#ORIGNAL FUNCTION: Compute grid output for each input row
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
trainingPercent = 0.9
Ntrain = math.floor(N * 0.9)
Ntest = N - Ntrain
trainIn = inArray[1:Ntrain,:]
trainOut = outArray3d[:,:,1:Ntrain]
testIn = inArray[Ntrain+1:,:]
testOut = outArray3d[:,:,Ntrain+1:]

#Then train & generate approximator outputs for different generators
#PLACEHOLDER: Output just dummy approximator
approximators = [
    (DummyApproxGenerator(inputFuncSource), 'dummy')
  ]
approx_output_dir = './approx/'
approx_files = []
for approx_info in approximators:
  approx = approx_info[0]
  approx_name = approx_info[1]
  approx_file = approx_name + '.c'
  approx.train(trainIn, trainOut)
  approx.generate(approx_output_dir, approx_file)
  approx_files.append(approx_output_dir + approx_file)

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
    # approx_func(inputRow, inputRow.size, approx_output, out_rows, out_cols)
    error = LA.norm(approx_output - orig_output)
    approx_errors_by_test[inputIdx] = error
  approx_errors.append(np.average(approx_errors_by_test))

print('AVERAGE APPROX ERRORS:')
for approxIdx, approx_error in enumerate(approx_errors):
  approx_name = approximators[approxIdx][1]
  print('  %s: %f' % (approx_name, approx_error))
