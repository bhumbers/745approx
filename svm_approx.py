import sys
sys.path.append('libraries/libsvm/python')

from svmutil import *

from approx_generator import ApproxGenerator

class SVMApproxGenerator(ApproxGenerator):
    def __init__(self, params):
        pass

    def typename(self):
        return 'svm'

    def train(self, inputs, outputs, params):
        self.params = params
        self.p = inputs.shape[1]
        self.n_r = outputs.shape[1]    #size of output grid in rows
        self.n_c = outputs.shape[2]    #size of output grid in cols

        self.in_min = inputs.min()
        self.in_scale = inputs.max() - self.in_min
        inputs = (inputs - self.in_min) / self.in_scale
        self.out_min = outputs.min()
        self.out_scale = outputs.max() - self.out_min
        outputs = (outputs - self.out_min) / self.out_scale

        for r in xrange(self.n_r):
            for c in xrange(self.n_c):
                x = map(list, inputs)
                y = list(outputs[:,r,c])

                # print x[:20]
                # print y[:20]
                # import cPickle as pickle
                # with open('%03d_%03d.dat' % (r, c), 'w') as f:
                #     pickle.dump([x, y], f)

                m = svm_train(y, x, '-s 3 -t 2 -c %(C)f -g %(g)f -p .001 -q' % params)
                svm_save_model('%03d_%03d_%03d.svm' % (params['index'], r, c), m)

    def generate(self, out_path, out_file, out_func_name):
        with open(path.join(out_path, out_file), 'w') as f:
            print >>f, '''
#include <stdio.h>
#include "svm.h"
static const int p = %(p)d, n_r = %(n_r)d, n_c = %(n_c)d;
static const double in_min = %(in_min).10f, in_scale = %(in_scale).10f;
static const double out_min = %(out_min).10f, out_scale = %(out_scale).10f;
void %(func)s(double input[p], int inputLen, double output[n_r][n_c], int _r, int _c){
  char fn[100];
  struct svm_model *m;
  struct svm_node in[p+1];
  //printf("in:  ");
  for(int i = 0; i < p; i++){
    //printf("%%5.2f ", input[i]);
    in[i].index = i+1;
    in[i].value = (input[i] - in_min) / in_scale;
  }
  //puts("");
  //printf("out: ");
  in[p].index = -1;
  for(int r = 0; r < n_r; r++){
    for(int c = 0; c < n_c; c++){
      sprintf(fn, "%(index)03d_%%03d_%%03d.svm", r, c);
      //puts(fn);
      m = svm_load_model(fn);

      output[r][c] = svm_predict(m, in) * out_scale + out_min;
      //printf("%%5.2f ", output[r][c]);
      svm_free_model_content(m);
      svm_free_and_destroy_model(&m);
    }
  }
  //puts("");
}
''' % dict(func=out_func_name, p=self.p, n_r=self.n_r, n_c=self.n_c,
           in_min=self.in_min, in_scale=self.in_scale,
           out_min=self.out_min, out_scale=self.out_scale, index=self.params['index'])
