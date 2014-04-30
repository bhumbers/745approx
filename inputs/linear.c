void linear_single(double* input, int inputLen, double* output, int outputRows, int outputCols)
{
  for(int row = 0; row < outputRows; row++) {
    for(int col = 0; col < outputCols; col++) {
      double x = col/(double)(outputCols);
      double y = row/(double)(outputRows);
      output[row*outputCols + col] = input[0] * x + input[1] * y;
    }
  }
}

void linear(int n_inst, double* input, int inputLen, double* output, int outputRows, int outputCols)
{
  for(int n = 0; n < n_inst; n++) {
    linear_single(input + n*inputLen, inputLen,
                  output + n * outputRows * outputCols,
                  outputRows, outputCols);
  }
}
