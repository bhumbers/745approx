#!/bin/bash

if [ -z "$(which swig)" ]; then
    echo "Must install SWIG!"
    exit
fi
if [ -z "$(which cmake)" ]; then
    echo "Must install cmake!"
    exit
fi

mkdir -p libraries/tmp_fann

cd libraries/tmp_fann
rm fann*/ FANN*/ -rf

## get package files
wget -nc http://downloads.sourceforge.net/project/fann/fann/2.1.0beta/fann-2.1.0beta.zip
wget -nc http://downloads.sourceforge.net/project/fann/fann/2.2.0/FANN-2.2.0-Source.zip
unzip fann-2.1.0beta.zip
unzip FANN-2.2.0-Source.zip

## move 2.1.0 python bindings to 2.2.0 directory
cp -rp fann-2.1.0/python FANN-2.2.0-Source/
rm fann-2.1.0 -r

## compile main package
cd FANN-2.2.0-Source
cmake .
make

## fix some things and compile python bindings
cd python
sed -i 's|\r||g' setup.py
sed -i 's|== dim|!= dim|' pyfann/pyfann.i
sed -i 's|../src/doublefann.o|doublefann|' setup.py
sed -i 's|extra_objects|libraries|' setup.py
sed -i '/ext_modules=/alibrary_dirs=["../src"], runtime_library_dirs=["$ORIGIN"],' setup.py
./setup.py build

## copy resulting files out
cd ..
cp -rp python/build/lib*/pyfann ../../
cp -rpL src/libdoublefann.so.2 ../../pyfann/
ln -sf libdoublefann.so.2 ../../pyfann/libdoublefann.so
cp -rp src/include ../../pyfann/

## test final result
cd ../..
echo -e '\nTesting import...'

if python -c 'import pyfann'; then
    echo 'Success!'
    #rm -r tmp_fann
fi
