#!/usr/bin/env python

from ctypes import *
import numpy as np
import ioutils

from dummy_approx import DummyApproxGenerator

#Script to execute approximation function generation for a given C/C++ function (in a shared lib)

saveFuncResults = True
resultsDir = './results'

#TODO: Read JSON config with
# - Input function(s) to approximate
#   - Output file names (if provided, else use defaults)
#   - Input generators of some sort...
# - Approximation config parameters (which to try, what params to use for each)
#PLACEHOLDER: Basic lib loading & execution (run "./inputs/make" before this)
inputLibFile = "./inputs/basic.so"
inputFunc = "basic_example"
inputFuncSource = "./inputs/basic.c" #only needed for DummyApproxGenerator

#PLACEHOLDER: Generate some arbitrary inputs (one row per call)
numInputTests = 5   # number of distinct input instances to test
inputLen = 3        # length of each input array
numOutRows = 3     # Size in rows of grid output
numOutCols = 3     # Size in cols of grid output
funcInputs = np.zeros([numInputTests, inputLen])
for i in range(numInputTests):
  for j in range(inputLen):
    funcInputs[i, j] = i
funcOutput = np.zeros([numOutRows, numOutCols])

if (saveFuncResults):
  ioutils.mkdir_p(resultsDir)

inputLib = CDLL(inputLibFile)

#Define the C function interface nicely
#Source: http://stackoverflow.com/questions/5862915/passing-np-arrays-to-a-c-function-for-input-and-output
func = inputLib[inputFunc]
func.restype = None
func.argtypes = [np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
                 c_int,
                 np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
                 c_int,
                 c_int]

#ORIGNAL FUNCTION: Compute grid output for each input row
for testIdx, inputRow in enumerate(funcInputs):
  funcOutput[:] = 0   #reset output before executing this call
  func(inputRow, inputRow.size, funcOutput, numOutRows, numOutCols)
  print(('Input #%d: \n' + str(inputRow) + '\n') % testIdx)
  print('Output: \n' + str(funcOutput) + '\n')
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

#TODO: Split data into training & test sets

#Then train & generate outputs for different generators
#PLACEHOLDER: Output just dummy generator
approximators = [DummyApproxGenerator(inputFuncSource)]
for approx in approximators:
  approx.train(inArray, outList)
  approx.generate('./approx')

# TODO: For each approximator, evaluate approximator quality on test set, save or show to console


