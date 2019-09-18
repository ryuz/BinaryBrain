// --------------------------------------------------------------------------
//  Binary Brain  -- binary neural net framework
//
//                                Copyright (C) 2018-2019 by Ryuji Fuchikami
//                                https://github.com/ryuz
//                                ryuji.fuchikami@nifty.com
// --------------------------------------------------------------------------


#include "Manager.cu"
#include "LocalHeap.cu"
#include "FrameBufferCopy.cu"
#include "Vector.cu"
#include "MatrixColwiseSum.cu"
#include "MatrixColwiseMeanVar.cu"
#include "MatrixRowwiseSetVector.cu"
#include "MicroMlp.cu"
#include "BinaryLut6.cu"
#include "SparseLut.cu"
#include "StochasticLut.cu"
#include "StochasticMaxPooling.cu"
#include "StochasticBatchNormalization.cu"
#include "ShuffleModulation.cu"
#include "BinaryToReal.cu"
#include "Im2Col.cu"
#include "Col2Im.cu"
#include "MaxPooling.cu"
#include "UpSampling.cu"
#include "BatchNormalization.cu"
#include "ReLU.cu"
#include "Sigmoid.cu"
#include "Binarize.cu"
#include "HardTanh.cu"
#include "Adam.cu"
#include "LossSoftmaxCrossEntropy.cu"
#include "AccuracyCategoricalClassification.cu"


// end of file
