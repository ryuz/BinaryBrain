import  os
from os.path import join as pjoin
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
from setuptools import setup, find_packages

__version__ = '0.0.2'


def find_in_path(name, path):
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def search_cuda_path():
    if sys.platform.startswith('win32') and 'CUDA_PATH' in os.environ:
        cuda_home    = os.environ['CUDA_PATH']
        cuda_bin     = pjoin(cuda_home, 'bin')
        cuda_include = pjoin(cuda_home, 'include')
        cuda_lib     = pjoin(cuda_home, 'lib', 'x64')
        cuda_nvcc    = pjoin(cuda_bin, 'nvcc')
    elif 'CUDAHOME' in os.environ:
        cuda_home = os.environ['CUDAHOME']
        cuda_bin     = pjoin(cuda_home, 'bin')
        cuda_include = pjoin(cuda_home, 'include')
        cuda_lib     = pjoin(cuda_home, 'lib64')
        cuda_nvcc    = pjoin(cuda_bin, 'nvcc')
    else:
        cuda_nvcc = find_in_path('nvcc', os.environ['PATH'])
        if cuda_nvcc is None:
            return None
        cuda_home = os.path.dirname(os.path.dirname(cuda_nvcc))
        cuda_bin     = pjoin(cuda_home, 'bin')
        cuda_include = pjoin(cuda_home, 'include')
        cuda_lib     = pjoin(cuda_home, 'lib64')

    return {'home':cuda_home, 'nvcc':cuda_nvcc, 'include': cuda_include, 'lib': cuda_lib}

cuda_path = search_cuda_path()


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)



# files
sources       = ['src/core.cpp']
define_macros = [('BB_WITH_CEREAL', '1')]
include_dirs  = [get_pybind_include(), get_pybind_include(user=True), '../include', '../cereal/include']
lib_dirs      = []

if cuda_path is not None:
    sources       += ['bbcu_thrust.cu']
    define_macros += [('BB_WITH_CUDA', '1')]
    include_dirs  += [cuda_path['include'], '../cuda']
    lib_dirs      += [cuda_path['lib']]


ext_modules = [
    Extension(
        'binarybrain.core',
        sources,
        define_macros=define_macros,
        include_dirs=include_dirs,
        language='c++'
    ),
]



def customize_compiler_for_nvcc(self):
    print('compiler_type =', self.compiler_type)
    self.src_extensions.append('.cu')
#   default_compiler_so = self.compiler_so
    super = self._compile
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_so', cuda_path['nvcc'])
            postargs = extra_postargs['nvcc']
#        else:
#            postargs = extra_postargs['gcc']
        super(obj, src, ext, cc_args, postargs, pp_opts)
#       self.compiler_so = default_compiler_so
    self._compile = _compile



# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag): return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/arch:AVX2', '/openmp', '/std:c++14', '/wd\"4819\"'],
        'unix': ['-mavx2', '-mfma', '-fopenmp'],
    }
    l_opts = {
        'msvc': [],
        'unix': ['-fopenmp'],
    }
    
    
    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)

setup(
    name='binarybrain',
    version=__version__,
    author='Ryuji Fuchikami',
    author_email='ryuji.fuchikami@nifty.com',
    url='https://github.com/ryuz/BinaryBrain',
    description='BinaryBrain for Python',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.3'],
    setup_requires=['pybind11>=2.3'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    packages=['binarybrain']
)
