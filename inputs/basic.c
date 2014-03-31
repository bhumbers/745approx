#include <stdio.h>

/**
Basic structure of an input functino for approximation.
Inputs should provided in input array with length specified by inputLen
Outputs should be placed in 2D row-major grid output array, with dimensions outputLen x outputCols
*/
void basic_example(double* input, int inputLen, double* output, int outputRows, int outputCols)
{
  //Put some arbitrary output based on inputs
  for (int i = 0; i < inputLen; i++) {
    for (int row = 0; row < outputRows; row++) {
      for (int col = 0; col < outputCols; col++) {
        double val = input[i] + row + col;
        output[row*outputCols + col] = val;
      }
    }
  }
}

void blah() {
  puts("Blah!");
}
