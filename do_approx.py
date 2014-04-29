#!/usr/bin/env python

import glob
import itertools
import os
import random
import shutil
import time
import timeit
import subprocess

from collections import namedtuple
from ctypes import *
from os.path import join

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from pylab import imread, mean
import math

import func_compiler

ApproxConfig = namedtuple('ApproxConfig', ['module', 'gen_class', 'name', 'params'])
EvalResult = namedtuple('EvalResult', ['name', 'hyperp', 'rms_error', 'grad_error', 'train_time', 'run_time', 'calls'])

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

def generate_median_filter_inputs(kernel_half_w, kernel_half_h, img_chunk_w, img_chunk_h):
    img = mean(imread('./imgs/edison_noise50.png'),2)
    (img_h, img_w) = img.shape

    #Size of little chunks that we'll cut the image into
    img_chunk_full_w = img_chunk_w + 2*kernel_half_w
    img_chunk_full_h = img_chunk_h + 2*kernel_half_h
    num_inputs = int(math.floor(img_w / img_chunk_w) * math.floor(img_h / img_chunk_h))
    input_length = 2 + (img_chunk_full_w*img_chunk_full_h)
    func_inputs = np.zeros([num_inputs, input_length])
    i = 0
    #Break the image into overlapping chunks to be consumed by the filter
    #Overlapping because a (kernel_half_w, kernerl_half_h) border region must be included per chunk as input to the filter
    for y in xrange(kernel_half_h, img_h - img_chunk_h - kernel_half_h, img_chunk_h):
        for x in xrange(kernel_half_w, img_w - img_chunk_w - kernel_half_w, img_chunk_w):
            img_chunk = img[y-kernel_half_h:(y+img_chunk_h+kernel_half_h), x-kernel_half_w:(x+img_chunk_w+kernel_half_w)]
            func_inputs[i, 0] = kernel_half_w
            func_inputs[i,1] = kernel_half_h
            func_inputs[i,2:] = np.reshape(img_chunk, img_chunk_full_w * img_chunk_full_h, 1)
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

#Train & write C function approximators of named function & return a list of
# dict objs giving approx gen info, including generated approximator source file names under the key 'file'
def generate_approximators(approx_config, func_name, trainIn, trainOut, approx_out_dir):
    approx_module = __import__(approx_config.module)
    approx_gen_class = getattr(approx_module, approx_config.gen_class)
    approx_gen = approx_gen_class(approx_config.params)
    
    approx_info = {}
    approx_info['files'] = []
    approx_info['hyperps'] = []
    approx_info['train_times'] = []
    for index, p in enumerate(approx_config.params['hyperp']):
        approx_gen.train(trainIn, trainOut, p)

        approx_out_file = approx_config.name + str(index) + '.c'
        t0 = time.time()
        approx_gen.generate(approx_out_dir, approx_out_file, func_name)
        t1 = time.time()
        approx_info['files'].append(join(approx_out_dir, approx_out_file))
        approx_info['hyperps'].append(p)
        approx_info['train_times'].append(t1 - t0)

    return approx_info


def lli_compile(func_source, func_name, inputs, in_size, out_r, out_c):
    shutil.copy2(func_source, 'temp_main.c')
    with open('temp_main.c', 'a') as f:
        f.write('int main() {\n')
        f.write('    double inputs[%d][%d] = {\n'
                % (inputs.shape[0], inputs.shape[1]) )
        for input_row in inputs:
            f.write('        {')
            for v in input_row:
                f.write('%f, ' % v)
            f.write('},\n')
        f.write('    };\n')
        f.write('    double output[%d][%d];\n\n' % (out_r, out_c) )
        f.write('    for (int i = 0; i < %d; i++) {\n' % inputs.shape[0])
        f.write('        %s(inputs[i], %d, (void*)output, %d, %d);\n'
                % (func_name, in_size, out_r, out_c) )
        f.write("    }\n");
        f.write("}\n");


def get_approx_num_calls():
    num_callgrind_calls = 0

    args = '-Ilibraries/pyfann/include temp_main.c -Llibraries/pyfann -std=c99 -ldoublefann -lm'
    subprocess.check_call(['gcc'] + args.split())

    print("Running valgrind (may take a while)...")
    args = '--tool=callgrind --log-file=out.txt ./a.out'
    callgrind_output = subprocess.check_output(['valgrind'] + args.split())

    # open output and parse to get calls
    with open('out.txt', 'r') as f:
        for line in f:
            if (line.find('Collected') > -1):
                line_tokens = line.split()
                num_callgrind_calls = line_tokens[3]

    ## Old code, for lli/llvm
    #args = '-emit-llvm -O3 -o out -c'.split() + ['temp_main.c']
    #subprocess.check_call(['clang'] + args)
    #args = '-stats -force-interpreter out'
    #head_output = subprocess.check_output(['lli'] + args.split())
    #print(head_output)
    return num_callgrind_calls



def evaluate_approx(approx_info, func_name, test_in, test_out):
    #For each approximator, evaluate approximator quality on test set, save or show to console

    n_r, n_c = test_out[0].shape
    best_rmse = np.inf
    best_index = 0

    for index, p in enumerate(approx_info['hyperps']):
        approx_file = approx_info['files'][index]
        approx_outputs = np.zeros(test_out.shape)
        approx_func = func_compiler.compile(approx_file, func_name)

        for input_row, output_row in zip(test_in, approx_outputs):
            approx_func(input_row, input_row.size, output_row, n_r, n_c)
        rmse = np.sqrt(mean_squared_error(test_out, approx_outputs))
        if (rmse < best_rmse):
            best_rmse = rmse
            best_index = index

    best_approx_file = approx_info['files'][best_index]

    #Collect metrics on best approximator
    #Find error of each approximation output compared to the original output
    t0 = time.time()
    best_approx_func = func_compiler.compile(best_approx_file, func_name)
    for input_row, output_row in zip(test_in, approx_outputs):
        best_approx_func(input_row, input_row.size, output_row, n_r, n_c)
    run_time = time.time() - t0
    _, test_grad_rows, test_grad_cols = np.gradient(test_out)
    _, approx_grad_rows, approx_grad_cols = np.gradient(approx_outputs)
    test_grad = np.array([test_grad_rows, test_grad_cols])
    approx_grad = np.array([approx_grad_rows, approx_grad_cols])
    rms_error = np.sqrt(mean_squared_error(test_out, approx_outputs))
    grad_error = np.sqrt(mean_squared_error(test_grad, approx_grad))
    train_time = approx_info['train_times'][best_index]
    outputs = approx_outputs

    num_inputs = test_in.shape[0]
    k = np.ceil(num_inputs / 20)
    samples = random.sample(xrange(num_inputs), int(k))
    sample_in = test_in[samples]
    lli_compile(best_approx_file, func_name, sample_in, np.size(test_in[0]), n_r, n_c)
    calls = get_approx_num_calls()

    return EvalResult(best_approx_file, approx_info['hyperps'][best_index], rms_error, grad_error, train_time, run_time, calls), outputs


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
        kernel_half_w = 3
        kernel_half_h = 3
        img_chunk_w = 30
        img_chunk_h = 30
        input_gen = lambda: generate_median_filter_inputs(kernel_half_w, kernel_half_h, img_chunk_w, img_chunk_h)


    #Generate some random sum-of-Gaussians inputs
    print 'Input generation...'
    func_inputs = input_gen()

    #ORIGNAL FUNCTION: Compute grid output for each input row
    print 'Computing original function results...'
    # Set the grid output size
    #We have to use a specific hardcoded size for the median filter domain (size of the test image in medianfilter.c)
    if input_type == MED_INPUT:
        out_rows = img_chunk_w
        out_cols = img_chunk_h
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

    #Generate list of hyperparameter tuples to be evaluated
    linear_hyperparams = [1]
    neural_hyperparams = [1] # Danny TODO
    svm_hyperparams = [1] # Danny TODO

    #Set up approximator configs of interest
    approx_configs = [ApproxConfig('dummy_approx', 'DummyApproxGenerator', 'dummy', {'src': func_source, 'hyperp': [1]}),
                      ApproxConfig('linear_approx', 'LinearApproxGenerator', 'linear', {'hyperp': linear_hyperparams}),
                      ApproxConfig('neural_approx', 'NeuralNetApproxGenerator', 'neural', {'hyperp': neural_hyperparams}),
                     ]

    #Then train & generate approximator outputs for different generators
    eval_results = []
    outputs = []
    approx_out_dir = './approx/'
    for config in approx_configs:
        approx_info = generate_approximators(config, func_name, trainIn, trainOut, approx_out_dir)
        eval_result, output = evaluate_approx(approx_info, func_name, testIn, testOut)
        eval_results.append(eval_result)
        outputs.append(output)

    print 'RESULTS BY APPROXIMATOR:'
    print 'Name, RMSE, Gradient RMSE, Train time, Avg runtime, Calls'
    for result in eval_results:
        print '%20s: %6.4f %6.4f %6.4f %6.4f %d' % (result.name, result.rms_error, result.grad_error, result.train_time, result.run_time, int(result.calls))

    print 'DONE'

    normalize_plot_range = True#(input_type is not MED_INPUT) #MED already has output in 0-255 range, so don't normalize
    plot_results(approx_configs, outputs, normalize_plot_range)
