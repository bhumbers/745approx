
#Compile a shared lib for basic example, then copy to appropriate folders for use by python test harness
cd ./inputs
make
cp basic.so ../lib/
cp basic.h ../include/
echo "Done with basic example prep"