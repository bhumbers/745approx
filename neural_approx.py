import ctypes
import os
import sys

from os import path

sys.path.append('libraries')
from pyfann import libfann

from approx_generator import ApproxGenerator

class NeuralNetApproxGenerator(ApproxGenerator):
    def __init__(self, params):
        pass

    def typename(self):
        return "neural"

    def train(self, inputs, outputs):
        self.p = inputs.shape[1]       #number of input features
        self.n_r = outputs.shape[1]    #size of output grid in rows
        self.n_c = outputs.shape[2]    #size of output grid in cols

        self.out_min = outputs.min()
        self.out_max = outputs.max()

        d = self.out_max - self.out_min
        self.out_min -= d / 98
        self.out_max -= d / 98

        outputs = (outputs - self.out_min) / (self.out_max - self.out_min)

        assert inputs.shape[0] == outputs.shape[0]

        nn = libfann.neural_net()
        #nn.create_standard_array((self.p, 50, 50, self.n_r*self.n_c))
        nn.create_shortcut_array((self.p, self.n_r*self.n_c))
        nn.set_learning_rate(.7)
        nn.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)
        nn.set_activation_function_output(libfann.SIGMOID)

        data = libfann.training_data()
        data.set_train_data(inputs, outputs.reshape((-1, self.n_r*self.n_c)))

        #nn.train_on_data(data, 500, 10, .001)
        nn.cascadetrain_on_data(data, 20, 1, .004)

        nn.save('nn.net')
        nn.destroy()

    def generate(self, out_path, out_file, out_func_name):
        with open(path.join(out_path, out_file), 'w') as f:
            print >>f, '''
#include <string.h>
#include "doublefann.h"
static const int p = %(p)d, n_r = %(n_r)d, n_c = %(n_c)d;
static const double A = %(A)f, B = %(B)f;

void %(func)s(double input[p], int inputLen, double output[n_r][n_c], int n_r, int n_c){
  struct fann *nn = fann_create_from_file("nn.net");
  double *nn_out = fann_run(nn, input);

  for(int i = 0; i < n_r; i++)
    for(int j = 0; j < n_c; j++)
      output[i][j] = nn_out[i*n_c+j] * B + A;
}
''' % dict(func=out_func_name, p=self.p, n_r=self.n_r, n_c=self.n_c,
           A=self.out_min, B=(self.out_max - self.out_min))

if __name__ == '__main__':
    nn = libfann.neural_net()
    nn.create_standard_array([3, 3, 3])
    nn.set_learning_rate(.7)
    nn.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)
    nn.set_activation_function_output(libfann.SIGMOID_SYMMETRIC)

    for _ in range(400):
        nn.train([1, -1, 1], [1, 1, 1])

    nn.print_connections()

    print nn.run([1, -1, 1])

    nn.save('nn.net')
    nn.destroy()
