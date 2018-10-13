
TARGET = mnist-sample

#CC     = g++
CC     = clang++
CFLAGS = -mavx2 -std=c++14 -I../../include -I../../../cereal/include -I../../../eigen

SRCS = main.cpp

HDRS = ../../include/bb/LoadCifar10.h                                  \
       ../../include/bb/LoadMnist.h                                    \
       ../../include/bb/NeuralNet.h                                    \
       ../../include/bb/NeuralNetAccuracyCategoricalClassification.h   \
       ../../include/bb/NeuralNetAccuracyFunction.h                    \
       ../../include/bb/NeuralNetAffine.h                              \
       ../../include/bb/NeuralNetBatchNormalization.h                  \
       ../../include/bb/NeuralNetBatchNormalizationAvx.h               \
       ../../include/bb/NeuralNetBatchNormalizationEigen.h             \
       ../../include/bb/NeuralNetBinarize.h                            \
       ../../include/bb/NeuralNetBinaryFilter.h                        \
       ../../include/bb/NeuralNetBinaryLut.h                           \
       ../../include/bb/NeuralNetBinaryLut6.h                          \
       ../../include/bb/NeuralNetBinaryLut6VerilogXilinx.h             \
       ../../include/bb/NeuralNetBinaryLutN.h                          \
       ../../include/bb/NeuralNetBinaryMultiplex.h                     \
       ../../include/bb/NeuralNetBinaryToReal.h                        \
       ../../include/bb/NeuralNetBuffer.h                              \
       ../../include/bb/NeuralNetConvCollapse.h                        \
       ../../include/bb/NeuralNetConvExpand.h                          \
       ../../include/bb/NeuralNetConvExpandM.h                         \
       ../../include/bb/NeuralNetConvolution.h                         \
       ../../include/bb/NeuralNetConvolutionPack.h                     \
       ../../include/bb/NeuralNetDenseAffineSigmoid.h                  \
       ../../include/bb/NeuralNetGroup.h                               \
       ../../include/bb/NeuralNetLayer.h                               \
       ../../include/bb/NeuralNetLayerBuf.h                            \
       ../../include/bb/NeuralNetLossCrossEntropyWithSoftmax.h         \
       ../../include/bb/NeuralNetLossFunction.h                        \
       ../../include/bb/NeuralNetMaxPooling.h                          \
       ../../include/bb/NeuralNetOptimizer.h                           \
       ../../include/bb/NeuralNetOptimizerAdam.h                       \
       ../../include/bb/NeuralNetOptimizerSgd.h                        \
       ../../include/bb/NeuralNetRealToBinary.h                        \
       ../../include/bb/NeuralNetReLU.h                                \
       ../../include/bb/NeuralNetSigmoid.h                             \
       ../../include/bb/NeuralNetSoftmax.h                             \
       ../../include/bb/NeuralNetSparseAffine.h                        \
       ../../include/bb/NeuralNetSparseAffineSigmoid.h                 \
       ../../include/bb/NeuralNetSparseBinaryAffine.h                  \
       ../../include/bb/NeuralNetSparseLayer.h                         \
       ../../include/bb/NeuralNetType.h                                \
       ../../include/bb/NeuralNetUtility.h                             \
       ../../include/bb/ShuffleSet.h                                   \
       ../../include/bb/TrainData.h



$TARGET: $(SRCS) $(HDRS)
	$(CC) -o $(TARGET) $(CFLAGS) $(SRCS)

all: mnist-sample

clean:
	rm -f mnist-sample


