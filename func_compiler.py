import ctypes
import subprocess
import tempfile

## takes a source file and a function name, and returns a callable corresponding
## to that function in the file
def compile(filename, funcname):
    f = tempfile.NamedTemporaryFile(suffix='.so')
    ## TODO: allow for specification of different options
    subprocess.check_call(['cc',
                           filename,
                           '-std=c99',
                           '-shared',
                           '-fPIC',
                           '-o', f.name,
                           ])
    return ctypes.CDLL(f.name)[funcname]

## takes a source code string and a function name, and returns a callable
## corresponding to that function in the code
def compile_string(code, funcname):
    src = tempfile.NamedTemporaryFile(suffix='.c')
    src.write(code)
    src.flush()
    return compile(src.name, funcname)
