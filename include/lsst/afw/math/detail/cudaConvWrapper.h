// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file
 *
 * @brief Set up for convolution, calls GPU convolution kernels
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

namespace lsst {
namespace afw {
namespace math {
namespace detail {

bool IsSufficientSharedMemoryAvailable_ForImgBlock(int filterW, int filterH, int pixSize);
bool IsSufficientSharedMemoryAvailable_ForImgAndMaskBlock(int filterW, int filterH, int pixSize);
bool IsSufficientSharedMemoryAvailable_ForSfn(int order, int kernelN);


enum SpatialFunctionType_t { sftChebyshev, sftPolynomial};

#ifdef GPU_BUILD
template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionImage_SpatiallyInvariantKernel(
        lsst::afw::gpu::detail::GpuBuffer2D<InPixelT>&    inImage,
        lsst::afw::gpu::detail::GpuBuffer2D<OutPixelT>&   outImage,
        lsst::afw::gpu::detail::GpuBuffer2D<KerPixel>&    kernel
                                                  );

template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionMI_SpatiallyInvariantKernel(
        lsst::afw::gpu::detail::GpuBuffer2D<InPixelT>&    inImageImg,
        lsst::afw::gpu::detail::GpuBuffer2D<VarPixel>&    inImageVar,
        lsst::afw::gpu::detail::GpuBuffer2D<MskPixel>&    inImageMsk,
        lsst::afw::gpu::detail::GpuBuffer2D<OutPixelT>&   outImageImg,
        lsst::afw::gpu::detail::GpuBuffer2D<VarPixel>&    outImageVar,
        lsst::afw::gpu::detail::GpuBuffer2D<MskPixel>&    outImageMsk,
        lsst::afw::gpu::detail::GpuBuffer2D<KerPixel>&    kernel
                                               );

template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionImage_LinearCombinationKernel(
        lsst::afw::gpu::detail::GpuBuffer2D<InPixelT>& inImage,
        std::vector<double> colPos,
        std::vector<double> rowPos,
        std::vector< lsst::afw::math::Kernel::SpatialFunctionPtr > sFn,
        lsst::afw::gpu::detail::GpuBuffer2D<OutPixelT>&                outImage,
        std::vector< lsst::afw::gpu::detail::GpuBuffer2D<KerPixel> >&  basisKernels,
        SpatialFunctionType_t sfType,
        bool doNormalize
                                                 );

template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionMI_LinearCombinationKernel(
        lsst::afw::gpu::detail::GpuBuffer2D<InPixelT>& inImageImg,
        lsst::afw::gpu::detail::GpuBuffer2D<VarPixel>& inImageVar,
        lsst::afw::gpu::detail::GpuBuffer2D<MskPixel>& inImageMsk,
        std::vector<double> colPos,
        std::vector<double> rowPos,
        std::vector< lsst::afw::math::Kernel::SpatialFunctionPtr > sFn,
        lsst::afw::gpu::detail::GpuBuffer2D<OutPixelT>&                outImageImg,
        lsst::afw::gpu::detail::GpuBuffer2D<VarPixel>&                 outImageVar,
        lsst::afw::gpu::detail::GpuBuffer2D<MskPixel>&                 outImageMsk,
        std::vector< lsst::afw::gpu::detail::GpuBuffer2D<KerPixel> >&  basisKernels,
        SpatialFunctionType_t sfType,
        bool doNormalize
                                              );

#endif //GPU_BUILD

}
}
}
} //namespace lsst::afw::math::detail ends



