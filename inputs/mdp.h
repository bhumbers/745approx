#pragma once

/**
Computes the value function for the MDP defined by inputs
Inputs should be a concatenation of reward specifiers, each of which has the parameters
x, y, and reward
(x,y) are defined over a normalized [0,1] x [0,1] grid space
*/
extern void compute_mdp_values(double* input, int inputLen, double* output, int outputRows, int outputCols);