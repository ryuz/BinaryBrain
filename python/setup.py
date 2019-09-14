
import  os
from os.path import join as pjoin
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
from setuptools import setup, find_packages
import subprocess
import urllib.request
import tarfile

#from distutils import ccompiler
#from distutils import unixccompiler
#from distutils import msvccompiler


__version__ = '0.0.3'



with urllib.request.urlopen('https://github.com/USCiLab/cereal/archive/v1.2.2.tar.gz') as r:
    with open('cereal.tar.gz', 'wb') as f:
        f.write(r.read())
with tarfile.open('./cereal.tar.gz', 'r') as tar:
    tar.extractall('.')


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
include_dirs  = [get_pybind_include(), get_pybind_include(user=True), 'src/include', 'cereal-1.2.2/include']
lib_dirs      = []

if CUDA is not None:
    sources       += ['src/core_bbcu.cu']
    define_macros += [('BB_WITH_CUDA', '1')]
    include_dirs  += [CUDA['include'], 'src/cuda']
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
        if CUDA is not None:
            args = [CUDA['nvcc'], '-shared', '-o', output_filename] + objects + extra_postargs
            print(' '.join(args))
            subprocess.call(args)
        else:
            super_link(target_desc, objects,
                output_filename, output_dir, libraries,
                library_dirs, runtime_library_dirs,
                export_symbols, debug, extra_preargs,
                extra_postargs, build_temp, target_lang)

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


package_data = {
    'binarybrain': [       
        'src/include/bb/Activation.h',
        'src/include/bb/Assert.h',
        'src/include/bb/AveragePooling.h',
        'src/include/bb/BackpropagatedBatchNormalization.h',
        'src/include/bb/BatchNormalization.h',
        'src/include/bb/Binarize.h',
        'src/include/bb/BinaryLutN.h',
        'src/include/bb/BinaryModulation.h',
        'src/include/bb/BinaryScaling.h',
        'src/include/bb/BinaryToReal.h',
        'src/include/bb/ConcatenateCoefficient.h',
        'src/include/bb/ConnectionTable.h',
        'src/include/bb/ConvolutionCol2Im.h',
        'src/include/bb/ConvolutionIm2Col.h',
        'src/include/bb/CudaUtility.h',
        'src/include/bb/DataAugmentationMnist.h',
        'src/include/bb/DataType.h',
        'src/include/bb/DenseAffine.h',
        'src/include/bb/Dropout.h',
        'src/include/bb/ExportVerilog.h',
        'src/include/bb/Filter2d.h',
        'src/include/bb/FixedSizeConnectionTable.h',
        'src/include/bb/FrameBuffer.h',
        'src/include/bb/HardTanh.h',
        'src/include/bb/LoadCifar10.h',
        'src/include/bb/LoadMnist.h',
        'src/include/bb/LoadXor.h',
        'src/include/bb/LossFunction.h',
        'src/include/bb/LossMeanSquaredError.h',
        'src/include/bb/LossSoftmaxCrossEntropy.h',
        'src/include/bb/LoweringConvolution.h',
        'src/include/bb/LutLayer.h',
        'src/include/bb/Manager.h',
        'src/include/bb/MaxPooling.h',
        'src/include/bb/Memory.h',
        'src/include/bb/MetricsBinaryAccuracy.h',
        'src/include/bb/MetricsCategoricalAccuracy.h',
        'src/include/bb/MetricsFunction.h',
        'src/include/bb/MetricsMeanSquaredError.h',
        'src/include/bb/MicroMlpAffine.h',
        'src/include/bb/MicroMlp.h',
        'src/include/bb/Model.h',
        'src/include/bb/NormalDistributionGenerator.h',
        'src/include/bb/OptimizerAdaGrad.h',
        'src/include/bb/OptimizerAdam.h',
        'src/include/bb/Optimizer.h',
        'src/include/bb/OptimizerSgd.h',
        'src/include/bb/PnmImage.h',
        'src/include/bb/RealToBinary.h',
        'src/include/bb/Reduce.h',
        'src/include/bb/ReLU.h',
        'src/include/bb/Runner.h',
        'src/include/bb/Sequential.h',
        'src/include/bb/ShuffleModulation.h',
        'src/include/bb/ShuffleSet.h',
        'src/include/bb/Sigmoid.h',
        'src/include/bb/SimdSupport.h',
        'src/include/bb/SparseBinaryLutN.h',
        'src/include/bb/SparseLayer.h',
        'src/include/bb/SparseLutDiscreteN.h',
        'src/include/bb/SparseLutN.h',
        'src/include/bb/StochasticBatchNormalization.h',
        'src/include/bb/StochasticLutN.h',
        'src/include/bb/StochasticLutSimd.h',
        'src/include/bb/StochasticMaxPooling2x2.h',
        'src/include/bb/StochasticMaxPooling.h',
        'src/include/bb/StochasticOperation.h',
        'src/include/bb/Tensor.h',
        'src/include/bb/TensorOperator.h',
        'src/include/bb/UniformDistributionGenerator.h',
        'src/include/bb/UpSampling.h',
        'src/include/bb/Utility.h',
        'src/include/bb/ValueGenerator.h',
        'src/include/bb/Variables.h',

        'src/include/bbcu/bbcu.h',
        'src/include/bbcu/bbcu_util.h',

        'src/cuda/AccuracyCategoricalClassification.cu',
        'src/cuda/Adam.cu',
        'src/cuda/BatchNormalization.cu',
        'src/cuda/Binarize.cu',
        'src/cuda/BinaryLut6.cu',
        'src/cuda/BinaryToReal.cu',
        'src/cuda/Col2Im.cu',
        'src/cuda/FrameBufferCopy.cu',
        'src/cuda/HardTanh.cu',
        'src/cuda/Im2Col.cu',
        'src/cuda/LocalHeap.cu',
        'src/cuda/LossSoftmaxCrossEntropy.cu',
        'src/cuda/Manager.cu',
        'src/cuda/MatrixColwiseMeanVar.cu',
        'src/cuda/MatrixColwiseSum.cu',
        'src/cuda/MatrixRowwiseSetVector.cu',
        'src/cuda/MaxPooling.cu',
        'src/cuda/MicroMlp.cu',
        'src/cuda/RealToBinary.cu',
        'src/cuda/ReLU.cu',
        'src/cuda/ShuffleModulation.cu',
        'src/cuda/Sigmoid.cu',
        'src/cuda/SparseLut.cu',
        'src/cuda/StochasticBatchNormalization.cu',
        'src/cuda/StochasticLut.cu',
        'src/cuda/StochasticMaxPooling.cu',
        'src/cuda/UpSampling.cu',
        'src/cuda/Vector.cu',
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
    install_requires=['pybind11>=2.3', 'tqdm'],
    setup_requires=['pybind11>=2.3'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    packages=['binarybrain'],
    package_data=package_data,
)

