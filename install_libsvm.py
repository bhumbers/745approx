#!/bin/bash
cd libraries
git clone --depth 5 https://github.com/cjlin1/libsvm
cd libsvm
make lib
ln -s libsvm.so.2 libsvm.so

cd python
if python -c 'import svmutil'; then
    echo 'Success!'
    #rm -r tmp_fann
fi
