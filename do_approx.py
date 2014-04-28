#!/usr/bin/env python

import glob
import itertools
import os
import random
import shutil
import time
import timeit

from collections import namedtuple
from ctypes import *
from os.path import join

import numpy as np
from sklearn.metrics import mean_squared_error

import func_compiler

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
            func_inputs[i, offset+2] = 0.1             #variance
            func_inputs[i, offset+3] = 0.1
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

#Generates a complete list of median filter problems where the filter kernel size varies between [0,0] and given [max_kernel_width, max_kernel_height]
#Note that this is a lot less interesting than having an actual 2D image as input, but that's a really high-dim problem
#that we're not going to mess with for this project.... the medianfilter function just uses a hardcoded test image.
def generate_median_filter_inputs(max_kernel_width, max_kernel_height):
    num_inputs = (max_kernel_width+1)*(max_kernel_height+1)
    input_length =   4 # length of each input array
    func_inputs = np.zeros([num_inputs, input_length])
    i = 0
    for kw in xrange(max_kernel_width+1):
        for kh in xrange(max_kernel_height+1):
            func_inputs[i, 0] = kw
            func_inputs[i, 1] = kh
            i += 1
    return func_inputs

def compute_func_results(func_source, func_name, func_inputs, out_rows, out_cols, results_dir):
    #Define the C function interface nicely
    #Source: http://stackoverflow.com/questions/5862915/passing-np-arrays-to-a-c-function-for-input-and-output
    orig_func = func_compiler.compile(func_source, func_name)

    save_results = True
    func_output = np.zeros([out_rows, out_cols])
    if save_results:
        shutil.rmtree(results_dir) #clear any existing results data
        os.makedirs(results_dir)
    for inputIdx, inputRow in enumerate(func_inputs):
        func_output[:] = 0   #reset output before executing this call
        orig_func(inputRow, inputRow.size, func_output, out_rows, out_cols)
        # print 'Input #%d:\n%s\n' % (testIdx, inputRow)
        # print 'Output:\n%s\n' % funcOutput
        if save_results:
            np.save(join(results_dir, '%s_%04d.in.npy' % (func_name, inputIdx)), inputRow)
            np.save(join(results_dir, '%s_%04d.out.npy' % (func_name, inputIdx)), func_output)
    # return func_output

def load_func_in_out_data(results_dir):
    #First, load all the training in/out data for the function
    inResFiles = sorted(glob.glob(join(results_dir, '*.in.npy')))
    outResFiles = sorted(glob.glob(join(results_dir, '*.out.npy')))

    assert len(inResFiles) == len(outResFiles)

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
        approx_module = __import__(approx_config.module)
        approx_gen_class = getattr(approx_module, approx_config.gen_class)
        approx_gen = approx_gen_class(approx_config.params)
        approx_gen.train(trainIn, trainOut)

        approx_out_file = approx_config.name + '.c'
        approx_gen.generate(approx_out_dir, approx_out_file, func_name)
        approx_files.append(join(approx_out_dir, approx_out_file))
    return approx_files

def evaluate_approx(approx_files, func_name, test_in, test_out):
    #For each approximator, evaluate approximator quality on test set, save or show to console
    errors = []
    outputs = []
    times = []

    n_r, n_c = test_out[0].shape
    for approx_file in approx_files:
        approx_outputs = np.zeros(test_out.shape)
        approx_func = func_compiler.compile(approx_file, func_name)

        #Find error of each approximation output compared to the original output
        t0 = time.time()
        for input_row, output_row in zip(test_in, approx_outputs):
            approx_func(input_row, input_row.size, output_row, n_r, n_c)
        times.append(time.time() - t0)
        errors.append(np.sqrt(mean_squared_error(test_out, approx_outputs)))
        outputs.append(approx_outputs)

    print 'RESULTS BY APPROXIMATOR:'
    for name, error, t in zip(approx_files, errors, times):
        print '%20s: %6.4f %6.4f' % (name, error, t)
    return errors, outputs

def plot_results(configs, outputs, normalize_range):
    try:
        import Image, math
    except ImportError: return

    N = min(100, outputs[0].shape[0], 100)
    W = int(math.ceil(math.sqrt(N)))
    H = (N + W-1) / W
    D = 20

    outputs = np.array([o[:N] for o in outputs])
    if normalize_range:
        a = outputs.min()
        b = outputs.max()
        outputs = (254 * (outputs - a) / (b - a)).astype(np.uint8)

    for o, c in zip(outputs, configs):
        img = Image.new('L', (W*out_cols*D, H*out_cols*D))
        for i, a in enumerate(o):
            img.paste(Image.fromarray(a).resize((out_cols * D, out_rows * D)),
                      (D * out_cols * (i % W), D * out_rows * (i / W)))
        img.save('out_%s.png' % c.name)

if __name__ == '__main__':
    #Script to execute approximation function generation for a given C/C++ function (in a shared lib)

    #TODO: Read JSON config with
    # - Input function(s) to approximate
    #   - Output file names (if provided, else use defaults)
    #   - Input generators of some sort...
    # - Approximation config parameters (which to try, what params to use for each)


    results_dir = './results'

    SOG_INPUT = 0
    MDP_INPUT = 1
    MED_INPUT = 2
    input_type = SOG_INPUT

    if input_type == SOG_INPUT:
        # Option #1: Sum-of-Gaussians
        num_inputs = 1000
        func_name = 'sum_of_gaussians'
        func_source = './inputs/gaussian.c'
        input_gen = lambda: generate_sum_of_gaussians_inputs(num_inputs, 3)

    elif input_type == MDP_INPUT:
        # Option #2: MDP
        num_inputs = 1000
        func_name = 'compute_mdp_values'  #"basic_example"
        func_source = './inputs/mdp.c' #"./inputs/basic.c"
        input_gen = lambda: generate_mdp_inputs(num_inputs, 1)

    elif input_type == MED_INPUT:
        # Option #3: Median filter on an image
        func_name = 'filter'
        func_source = './inputs/medianfilter.c'
        input_gen = lambda: generate_median_filter_inputs(5,5)

    #Set up approximator configs of interest
    approx_configs = [ApproxConfig('dummy_approx', 'DummyApproxGenerator', 'dummy', {'src': func_source}),
                      ApproxConfig('linear_approx', 'LinearApproxGenerator', 'linear', {}),
                      ApproxConfig('neural_approx', 'NeuralNetApproxGenerator', 'neural', {}),
                      ]

    #Generate some random sum-of-Gaussians inputs
    print 'Input generation...'
    func_inputs = input_gen()

    #ORIGNAL FUNCTION: Compute grid output for each input row
    print 'Computing original function results...'
    # Set the grid output size
    #We have to use a specific hardcoded size for the median filter domain (size of the test image in medianfilter.c)
    if input_type == MED_INPUT:
        out_rows = 100
        out_cols = 100
    #But modify the other problems as you see fit
    else:
        out_rows = 10
        out_cols = 10
    compute_func_results(func_source, func_name, func_inputs, out_rows, out_cols, results_dir)

    #APPROXIMATORS: Train & generate C function to replace original function
    # (NOTE: This part can & should be split out from in/out pair generation for original function)

    print 'Generator training...'
    #Split data into training & test sets
    trainingFrac = 0.5
    inArray, outArray = load_func_in_out_data(results_dir)
    N = inArray.shape[0]
    Ntrain = int(N * trainingFrac)
    trainIn = inArray[:Ntrain]
    trainOut = outArray[:Ntrain]
    testIn = inArray[Ntrain:]
    testOut = outArray[Ntrain:]

    #Then train & generate approximator outputs for different generators
    approx_out_dir = './approx/'
    approx_files = generate_approximators(approx_configs, func_name, trainIn, trainOut, approx_out_dir)

    print 'Evaluating...'
    _, outputs = evaluate_approx(approx_files, func_name, testIn, testOut)

    print 'DONE'

    normalize_plot_range = (input_type is not MED_INPUT) #MED already has output in 0-255 range, so don't normalize
    plot_results(approx_configs, outputs, normalize_plot_range)
