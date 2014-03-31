#!/usr/bin/env python

import numpy
from ctypes import *

#Script to execute approximation function generation for a C/C++ function

#PLACEHOLDER: Generate some arbitrary inputs (one row per call)
# argInputs = numpy.array([10, 10])

#TEST: Lib loading & execution
inputLib = CDLL("./inputs/basic.so")
inputLib.



#TODO: Read JSON config with
# - Input function files
#   - Output file names (if provided, else use defaults)
#   - Input generators of some sort...
# - Approximation config parameters (which to try, what params to use for each)


# TODO: For each input func:
#   - Compile
#   - Generate/marshall test inputs
#     - Just split CSV input into rows for now?
#   - Run function on each test input, save outputs (eg: to CSV)
#   - For each approximator:
#     - Train parameters on input/output pairs
#     - Write C code for approximator, save to file, compile
#     - Evaluate approximator quality on all inputs, save/show to user


