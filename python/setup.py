
import  os
from os.path import join as pjoin
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
from setuptools import setup, find_packages

from distutils import ccompiler
from distutils import unixccompiler
from distutils import msvccompiler


__version__ = '0.0.3'


def find_in_path(name, path):
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def search_cuda():
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

CUDA = search_cuda()
#CUDA = None


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
sources       = ['src/core_main.cpp']
define_macros = [('BB_WITH_CEREAL', '1')]
include_dirs  = [get_pybind_include(), get_pybind_include(user=True), '../include', '../cereal/include']
lib_dirs      = []

if CUDA is not None:
    sources       += ['src/core_bbcu.cu']
    define_macros += [('BB_WITH_CUDA', '1')]
    include_dirs  += [CUDA['include'], '../cuda']
    lib_dirs      += [CUDA['lib']]


ext_modules = [
    Extension(
        'binarybrain.core',
        sources,
        define_macros=define_macros,
        include_dirs=include_dirs,
        language='c++'
    ),
]


import subprocess

def hook_compiler(self):
    self.src_extensions.append('.cu')
    super_compile = self._compile
    super_link = self.link

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            postargs = extra_postargs['cu']
        elif os.path.splitext(src)[1] == '.cpp':
            postargs = extra_postargs['cc']
        else:
            postargs = []
        super_compile(obj, src, ext, cc_args, postargs, pp_opts)
    
    def link(target_desc, objects,
             output_filename, output_dir=None, libraries=None,
             library_dirs=None, runtime_library_dirs=None,
             export_symbols=None, debug=0, extra_preargs=None,
             extra_postargs=None, build_temp=None, target_lang=None):


        print('-----------------')
        print('target_desc=', target_desc)
        print('objects=', objects)
        print('output_filename=', output_filename)
        print('output_dir=', output_dir)
        print('libraries=', libraries)
        print('library_dirs=', library_dirs)
        print('runtime_library_dirs=', runtime_library_dirs)
        print('export_symbols=', export_symbols)
        print('debug=', debug)
        print('extra_preargs=', extra_preargs)
        print('extra_postargs=', extra_postargs)
        print('build_temp=', build_temp)
        print('target_lang=', target_lang)
        print('-----------------')

        if CUDA is not None:
            args = ['nvcc' , '-shared', '-o', output_filename] + objects + extra_postargs
            print(args)
            subprocess.call(args)
        else:
            super_link(target_desc, objects,
                output_filename, output_dir=output_dir, libraries=libraries,
                library_dirs=library_dirs, runtime_library_dirs=runtime_library_dirs,
                export_symbols=export_symbols, debug=debug, extra_preargs=extra_preargs,
                extra_postargs=extra_postargs, build_temp=build_temp, target_lang=target_lang)

    self._compile = _compile
    self.link = link


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    cc_args = {'unix':[], 'msvc':[]}
    cu_args = {'unix':[], 'msvc':[]}
    ar_args = {'unix':[], 'msvc':[]}
    if CUDA is None:
        # unix(cpu)
        cc_args['unix'] += ['-mavx2', '-mfma', '-fopenmp', '-std=c++14']
        ar_args['unix'] += ['-fopenmp', '-lstdc++', '-lm']

        # windows(cpu)
        cc_args['msvc'] += ['/EHsc', '/arch:AVX2', '/openmp', '/std:c++14']
        ar_args['msvc'] += []
    else:
        # unix(gpu)
        cc_args['unix'] += ['-gencode=arch=compute_35,code=sm_35',
                            '-gencode=arch=compute_75,code=sm_75',
                            '-Xcompiler', '-pthread',
                            '-Xcompiler', '-mavx2',
                            '-Xcompiler', '-mfma',
                            '-Xcompiler', '-fopenmp',
                            '-Xcompiler', '-std=c++14',
                            '-Xcompiler', '-fPIC' ]
        cu_args['unix'] += ['-gencode=arch=compute_35,code=sm_35',
                            '-gencode=arch=compute_75,code=sm_75',
                            '-std=c++11',
                            '-Xcompiler', '-fPIC' ]
        ar_args['unix'] += ['-Xcompiler', '-pthread',
                            '-Xcompiler', '-fopenmp',
                            '-lstdc++', '-lm', '-lcublas']

        # windows(gpu)
        cc_args['msvc'] += ['/EHsc', '/arch:AVX2', '/openmp', '/std:c++14', '/wd\"4819\"']
        cu_args['msvc'] += ['-std=c++11',
                            '-gencode=arch=compute_35,code=sm_35',
                            '-gencode=arch=compute_75,code=sm_75']
        ar_args['msvc'] += []
    
    if sys.platform == 'darwin':
        darwin_args = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        cc_args['unix'] += darwin_args
        ar_args['unix'] += darwin_args


    def build_extensions(self):
        if CUDA is not None:
            self.compiler.set_executable('compiler_so', CUDA['nvcc'])
            self.compiler.set_executable('compiler_cxx', CUDA['nvcc'])
        
        hook_compiler(self.compiler)

        ct = self.compiler.compiler_type
        for ext in self.extensions:
            ext.extra_compile_args = {'cc': self.cc_args[ct], 'cu': self.cu_args[ct]}
            ext.extra_link_args = self.ar_args[ct]
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
    install_requires=['pybind11>=2.3', 'tqdm'],
    setup_requires=['pybind11>=2.3'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    packages=['binarybrain']
)

