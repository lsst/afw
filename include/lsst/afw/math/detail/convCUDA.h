// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file
 *
 * @brief GPU convolution code
 *
 * The functions listed in this header file call GPU convolution kernels.
 * All data must be prepared and uploaded to GPU.
 * Results are placed in GPU global memory.
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

namespace lsst {
namespace afw {
namespace math {
namespace detail {

typedef lsst::afw::image::VariancePixel VarPixel;
typedef lsst::afw::image::MaskPixel     MskPixel;
typedef double KerPixel;

namespace gpu {

const int maxGpuSfCount=100;

#ifdef GPU_BUILD

// image block size per GPU block. (The size of the image that one GPU block processes)
#define blockSizeX 32
#define blockSizeY 16

template <typename T>
void CallTestGpuKernel(T* ret);

void Call_ChebyshevImageValues(
        double* out, int outW, int outH,
        int order,
        double* params,
        double* rowPos,
        double* colPos,
        double minX, double minY, double maxX, double maxY,
        int sharedMemorySize
);

void Call_PolynomialImageValues(
        double* out, int outW, int outH,
        int order,
        double* params,
        double* rowPos,
        double* colPos,
        int sharedMemorySize
);

void Call_NormalizationImageValues(
        double* out, int outW, int outH,
        double** sFn, int n,
        double* kernelSum,
        bool* isDivideByZeroGPU,
        int blockN,
        int sharedMemorySize
);

template <typename OutPixelT, typename InPixelT>
void Call_SpatiallyInvariantImageConvolutionKernel(
        InPixelT*  inImageGPU, int inImageWidth, int inImageHeight,
        KerPixel*  allKernelsGPU, int kernelTotalN,
        int kernelW, int kernelH,
        OutPixelT* outImageGPU[],
        int blockN,
        int sharedMemorySize
);

void Call_SpatiallyInvariantMaskConvolutionKernel(
        MskPixel*  inImageGPU, int inImageWidth, int inImageHeight,
        KerPixel*  allKernelsGPU, int kernelTotalN,
        int kernelW, int kernelH,
        MskPixel* outImageGPU[],
        int blockN,
        int sharedMemorySize
);

template <typename OutPixelT, typename InPixelT>
void Call_ConvolutionKernel_LC_Img(
        InPixelT*  inImageGPU, int inImageWidth, int inImageHeight,
        KerPixel*  kernelGPU, int kernelTotalN,
        int kernelW, int kernelH,
        double* sfValGPU[],
        double* normGPU,
        OutPixelT* outImageGPU,
        int blockN,
        int sharedMemorySize
);

void Call_ConvolutionKernel_LC_Var(
        VarPixel*  inImageGPU, int inImageWidth, int inImageHeight,
        MskPixel*  inMskGPU,
        KerPixel*  kernelGPU, int kernelTotalN,
        int kernelW, int kernelH,
        double*  sfValGPU[],
        double* normGPU,
        VarPixel* outImageGPU,
        MskPixel*  outMskGPU,
        int blockN,
        int sharedMemorySize
);

#endif //GPU_BUILD

}
}
}
}
}  //namespace lsst::afw::math::detail::gpu ends

