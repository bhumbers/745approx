#include <stdio.h>

#include <assert.h>
#include <math.h>

void sum_of_gaussians(double* input, int inputLen, double* output, int outputRows, int outputCols)
{
  //Inputs should be 5-tuples: the 2D mean, 2D variance, and amplitude per Gaussian
  assert(inputLen % 5 == 0);

  int numGaussians = inputLen / 5;
  for (int i = 0; i < numGaussians; i += 5) {
    for (int row = 0; row < outputRows; row++) {
      for (int col = 0; col < outputCols; col++) {
        double meanX =      input[i+0];
        double meanY =      input[i+1];
        double varX =       input[i+2];
        double varY =       input[i+3];
        double amplitude =  input[i+4];
        double x = col/(double)(outputCols);
        double y = row/(double)(outputRows);
        double val = amplitude * exp(-( (x - meanX)*(x - meanX)/(2*varX*varX) +
                                        (y - meanY)*(y - meanY)/(2*varY*varY)));
        output[row*outputCols + col] += val;
      }
    }
  }
}
