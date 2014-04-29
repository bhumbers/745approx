void linear(double* input, int inputLen, double* output, int outputRows, int outputCols)
{
  for(int row = 0; row < outputRows; row++) {
    for(int col = 0; col < outputCols; col++) {
        double x = col/(double)(outputCols);
        double y = row/(double)(outputRows);
        output[row*outputCols + col] = input[0] * x + input[1] * y;
    }
  }
}
