#pragma once

/** Applies a median filter to a statically defined input image, encoded as follows:
  input[0]: Half x-size of filter kernel window (halfKw) (integer); actual filter window is of width  (2*halfKw + 1)
  input[1]: Half y-size of filter kernel window (halfKh) (integer); actual filter window is of height (2*halfKh + 1)
  input[2...2+kw*kh]: 2D image section corresponding to the region to be filtered, with the target pixel to be updated at the center of the region
*/
extern void filter(double* input, int inputLen, double* output, int outputRows, int outputCols);
