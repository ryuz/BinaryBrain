// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#ifndef BB_PYBIND11
#define BB_PYBIND11
#endif

#ifndef BB_OBJECT_LOADER
#define BB_OBJECT_LOADER
#endif


#include "Manager.cu"
#include "LocalHeap.cu"
#include "FrameBufferCopy.cu"
#include "ConvBitToReal.cu"
#include "Vector.cu"
#include "MatrixColwiseSum.cu"
#include "MatrixColwiseMeanVar.cu"
#include "MatrixRowwiseSetVector.cu"
#include "MicroMlp.cu"
#include "AverageLut.cu"
#include "MaxLut.cu"
#include "BinaryLut6.cu"
#include "DifferentiableLut.cu"
#include "StochasticLut.cu"
#include "StochasticMaxPooling.cu"
#include "StochasticBatchNormalization.cu"
#include "ShuffleModulation.cu"
#include "Shuffle.cu"
#include "RealToBinary.cu"
#include "BinaryToReal.cu"
#include "BitEncode.cu"
#include "BitError.cu"
#include "Im2Col.cu"
#include "Col2Im.cu"
#include "MaxPooling.cu"
#include "UpSampling.cu"
#include "BatchNormalization.cu"
#include "ReLU.cu"
#include "Sigmoid.cu"
#include "Binarize.cu"
#include "HardTanh.cu"
#include "OptimizerAdam.cu"
#include "LossSoftmaxCrossEntropy.cu"
#include "LossMeanSquaredError.cu"
#include "MetricsCategoricalAccuracy.cu"
#include "Utility.cu"


// end of file
