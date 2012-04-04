// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
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



