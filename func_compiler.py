import ctypes
import subprocess
import tempfile

## takes a source file and a function name, and returns a callable corresponding
## to that function in the file
def compile(filename, funcname):
    return compile_files([filename], funcname)

## takes a list of source files and a function name, and returns a callable corresponding
## to that function in the file
def compile_files(filenames, funcname):
    f = tempfile.NamedTemporaryFile(suffix='.so')
    ## TODO: allow for specification of different options
    compile_cmd = ['cc']
    compile_cmd.extend(filenames)
    compile_cmd.extend(['-std=c99',
                        '-shared',
                        '-Llibraries/pyfann',
                        '-Ilibraries/pyfann/include',
                        '-ldoublefann',
                        '-fPIC',
                        '-o', f.name])
    subprocess.check_call(compile_cmd)
    return ctypes.CDLL(f.name)[funcname]


## takes a source code string and a function name, and returns a callable
## corresponding to that function in the code
def compile_string(code, funcname):
    src = tempfile.NamedTemporaryFile(suffix='.c')
    src.write(code)
    src.flush()
    return compile(src.name, funcname)
