""" setup.py for Binary Brain
"""

import sys
import os
from os.path import join as pjoin
import setuptools
from setuptools import setup, Extension
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import distutils
import subprocess
import urllib.request
import tarfile
import re

# from distutils import ccompiler
# from distutils import unixccompiler
# from distutils import msvccompiler


# change directory
src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(src_path)


# build flags
VERBOSE     = False
WITH_CUDA   = True
WITH_CEREAL = False


# version
major_version = 0
minor_version = 0
revision_number = 0
with open('binarybrain/include/bb/Version.h', 'r', encoding="utf-8") as f:
    for line in f.readlines():
        m = re.match(r'\s*#\s*define\s+BB_MAJOR_VERSION\s+([0-9]+)', line)
        if m:   major_version = int(m.group(1))
        m = re.match(r'\s*#\s*define\s+BB_MINOR_VERSION\s+([0-9]+)', line)
        if m:   minor_version = int(m.group(1))
        m = re.match(r'\s*#\s*define\s+BB_REVISION_NUMBER\s+([0-9]+)', line)
        if m:   revision_number = int(m.group(1))

__version__ = str(major_version) + '.' + str(minor_version) + '.' + str(revision_number)

if VERBOSE:
    print('version = %s' % __version__)




# wget cereal
if WITH_CEREAL:
    with urllib.request.urlopen('https://github.com/USCiLab/cereal/archive/v1.2.2.tar.gz') as r:
        with open('cereal.tar.gz', 'wb') as f:
            f.write(r.read())
    with tarfile.open('./cereal.tar.gz', 'r') as tar:
        tar.extractall('.')

# search CUDA
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

if WITH_CUDA:
    CUDA = search_cuda()
else:
    CUDA = None

if CUDA is None:
    print('CUDA is not found.')

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
sources       = ['binarybrain/src/core_main.cpp']
define_macros = [('BB_ASSERT_EXCEPTION', '1')]
include_dirs  = [distutils.sysconfig.get_python_inc(), get_pybind_include(), get_pybind_include(user=True), 'binarybrain/include']
lib_dirs      = []

if WITH_CEREAL:
    define_macros += [('BB_WITH_CEREAL', '1')]
    include_dirs  += ['cereal-1.2.2/include']

if CUDA is not None:
    sources       += ['binarybrain/src/core_bbcu.cu']
    define_macros += [('BB_WITH_CUDA', '1')]
    include_dirs  += [CUDA['include'], 'binarybrain/cuda']
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


def hook_compiler(self):
    self.src_extensions.append('.cu')
    super_compile_  = self._compile
    super_compile   = self.compile
    super_link      = self.link

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if VERBOSE:
            print('---------------------')
            print('[_compile]')
            print('obj =', obj)
            print('src =', src)
            print('ext =', ext)
            print('cc_args =', cc_args)
            print('extra_postargs =', extra_postargs)
            print('pp_opts =', pp_opts)
            print('---------------------')
        if os.path.splitext(src)[1] == '.cu':
            postargs = extra_postargs['cu']
        elif os.path.splitext(src)[1] == '.cpp':
            postargs = extra_postargs['cc']
        else:
            postargs = []
        super_compile_(obj, src, ext, cc_args, postargs, pp_opts)
    
    def compile(sources,
                output_dir=None, macros=None, include_dirs=None, debug=0,
                extra_preargs=None, extra_postargs=None, depends=None):
        
        if VERBOSE:
            print('---------------------')
            print('[compile]')
            print('sources =', sources)
            print('output_dir =', output_dir)
            print('macros =', macros)
            print('include_dirs =', include_dirs)
            print('debug =', debug)
            print('extra_preargs =', extra_preargs)
            print('extra_postargs =', extra_postargs)
            print('---------------------')

        if self.compiler_type == 'unix':
            return super_compile(sources,
                        output_dir, macros, include_dirs, debug,
                        extra_preargs, extra_postargs, depends)

        if CUDA is not None:
            macros, objects, extra_postargs, _, _ = \
            self._setup_compile(output_dir, macros, include_dirs,
                            sources, depends, extra_postargs)
            
            # macros
            macs = []
            for mac in macros:
                if len(mac) >= 2:
                    macs.append('-D' + mac[0] + '=' + mac[1])
                else:
                    macs.append('-D' + mac[0])

            # includes
            incs = []
            if self.compiler_type == 'msvc':
                incs += ['-I"' + str(inc) + '"' for inc in include_dirs]
            else:
                incs += ['-I' + str(inc) for inc in include_dirs]
            
            # compile
            objects = []
            for src in sources:
                postargs = []
                if os.path.splitext(src)[1] == '.cu':
                    postargs = extra_postargs['cu']
                elif os.path.splitext(src)[1] == '.cpp':
                    postargs = extra_postargs['cc']

                fname, _ = os.path.splitext(os.path.basename(src))
                obj = os.path.join(output_dir, fname + self.obj_extension)
                objects.append(obj)

                args = [CUDA['nvcc'], '-c', '-o', obj] + incs + macs + [src] + postargs
                print(' '.join(args))
                subprocess.call(args)
#               self.spawn(args)

            return objects
        else:
            return super_compile(sources,
                        output_dir, macros, include_dirs, debug,
                        extra_preargs, extra_postargs['cc'], depends)

    def link(target_desc, objects,
             output_filename, output_dir=None, libraries=None,
             library_dirs=None, runtime_library_dirs=None,
             export_symbols=None, debug=0, extra_preargs=None,
             extra_postargs=None, build_temp=None, target_lang=None):
        
        if VERBOSE:
            print('---------------------')
            print('[link]')
            print('target_desc =', target_desc)
            print('objects =', objects)
            print('libraries =', libraries)
            print('library_dirs =', library_dirs)
            print('runtime_library_dirs =', runtime_library_dirs)
            print('export_symbols =', export_symbols)
            print('debug =', debug)
            print('extra_preargs =', extra_preargs)
            print('extra_postargs =', extra_postargs)
            print('build_temp =', build_temp)
            print('target_lang =', target_lang)
            print('---------------------')

        if CUDA is not None:
            libraries, library_dirs, runtime_library_dirs =\
                    self._fix_lib_args(libraries, library_dirs, runtime_library_dirs)
            
#           os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            
            lib_dirs = []
            if self.compiler_type == 'msvc':
                lib_dirs += ['-L"' + str(libdir) + '"' for libdir in library_dirs]
            else:
                lib_dirs += ['-L' + str(libdir) for libdir in library_dirs]
            
            args = [CUDA['nvcc'], '-shared', '-o', output_filename] + objects + lib_dirs + extra_postargs
            print(' '.join(args))
            subprocess.call(args)
#           self.spawn(args)
        else:
            super_link(target_desc, objects,
                output_filename, output_dir, libraries,
                library_dirs, runtime_library_dirs,
                export_symbols, debug, extra_preargs,
                extra_postargs, build_temp, target_lang)

    # hook
    if self.compiler_type == 'unix':
        self._compile = _compile
    self.compile = compile
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
        cc_args['msvc'] += ['/EHsc', '/Oi', '/MT', '/arch:AVX2', '/openmp', '/std:c++14', '/wd"4819"']
        ar_args['msvc'] += []
    else:
        # unix(gpu)
        cc_args['unix'] += ['-gencode=arch=compute_35,code=sm_35',
                            '-gencode=arch=compute_50,code=sm_50',
                            '-gencode=arch=compute_60,code=sm_60',
                            '-gencode=arch=compute_61,code=sm_61',
                            '-gencode=arch=compute_75,code=sm_75',
                            '-Xcompiler', '-pthread',
                            '-Xcompiler', '-mavx2',
                            '-Xcompiler', '-mfma',
                            '-Xcompiler', '-fopenmp',
                            '-Xcompiler', '-std=c++14',
                            '-Xcompiler', '-fPIC' ]
        cu_args['unix'] += ['-gencode=arch=compute_35,code=sm_35',
                            '-gencode=arch=compute_50,code=sm_50',
                            '-gencode=arch=compute_60,code=sm_60',
                            '-gencode=arch=compute_61,code=sm_61',
                            '-gencode=arch=compute_75,code=sm_75',
                            '-std=c++11',
                            '-Xcompiler', '-fPIC' ]
        ar_args['unix'] += ['-Xcompiler', '-pthread',
                            '-Xcompiler', '-fopenmp',
                            '-lstdc++', '-lm', '-lcublas']
        
        # windows(gpu)
        cc_args['msvc'] += ['-O3',
                            '-Xcompiler', '/bigobj',
                            '-Xcompiler', '/EHsc',
                            '-Xcompiler', '/O2',
                            '-Xcompiler', '/Oi',
                            '-Xcompiler', '/FS',
                            '-Xcompiler', '/Zi',
                            '-Xcompiler', '/MT',
                            '-Xcompiler', '/arch:AVX2',
                            '-Xcompiler', '/openmp',
                            '-Xcompiler', '/std:c++14',
                            '-Xcompiler', '/wd\"4819\"']
        cu_args['msvc'] += ['-O3',
                            '-std=c++14',
                            '-gencode=arch=compute_35,code=sm_35',
                            '-gencode=arch=compute_50,code=sm_50',
                            '-gencode=arch=compute_60,code=sm_60',
                            '-gencode=arch=compute_61,code=sm_61',
                            '-gencode=arch=compute_75,code=sm_75',
                            '-Xcompiler', '/bigobj',
                            '-Xcompiler', '/EHsc',
                            '-Xcompiler', '/O2',
                            '-Xcompiler', '/Oi',
                            '-Xcompiler', '/FS',
                            '-Xcompiler', '/Zi',
                            '-Xcompiler', '/MT',
                            '-Xcompiler', '/wd\"4819\"']
        ar_args['msvc'] += ['-lcublas']
    
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

package_data = {
    'binarybrain': [
        'include/bb/*.h',
        'include/bbcu/*.h',
        'cuda/*.cu',
        'cuda/*.cuh',
    ],
}

setup(
    name='binarybrain',
    version=__version__,
    author='Ryuji Fuchikami',
    author_email='ryuji.fuchikami@nifty.com',
    url='https://github.com/ryuz/BinaryBrain',
    description='BinaryBrain for Python',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.4', 'numpy'],
    setup_requires=['pybind11>=2.4'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    packages=['binarybrain'],
    package_data=package_data,
)

