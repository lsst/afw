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

#ifdef GPU_BUILD

namespace lsst {
namespace afw {
namespace math {
namespace detail {
namespace gpu {

bool IsSufficientSharedMemoryAvailable_ForImgBlock(int filterW, int filterH, int pixSize);
bool IsSufficientSharedMemoryAvailable_ForImgAndMaskBlock(int filterW, int filterH, int pixSize);
bool IsSufficientSharedMemoryAvailable_ForSfn(int order, int kernelN);

// returns true when preffered device has been selected
// returns false when there is no preffered device
// throws exception when unable to select preffered device
bool SelectPreferredCudaDevice();

// throws exception when automatic selection fails
void AutoSelectCudaDevice();

// verifies basic parameters of Cuda device
void VerifyCudaDevice();

} //namespace gpu ends

enum SpatialFunctionType_t { sftChebyshev, sftPolynomial};

template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionImage_SpatiallyInvariantKernel(
    ImageBuffer<InPixelT>&    inImage,
    ImageBuffer<OutPixelT>&   outImage,
    ImageBuffer<KerPixel>&    kernel
);

template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionMI_SpatiallyInvariantKernel(
    ImageBuffer<InPixelT>&    inImageImg,
    ImageBuffer<VarPixel>&    inImageVar,
    ImageBuffer<MskPixel>&    inImageMsk,
    ImageBuffer<OutPixelT>&   outImageImg,
    ImageBuffer<VarPixel>&    outImageVar,
    ImageBuffer<MskPixel>&    outImageMsk,
    ImageBuffer<KerPixel>&    kernel
);

template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionImage_LinearCombinationKernel(
    ImageBuffer<InPixelT>& inImage,
    std::vector<double> colPos,
    std::vector<double> rowPos,
    std::vector< lsst::afw::math::Kernel::SpatialFunctionPtr > sFn,
    ImageBuffer<OutPixelT>&                outImage,
    std::vector< ImageBuffer<KerPixel> >&  basisKernels,
    SpatialFunctionType_t sfType,
    bool doNormalize
);

template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionMI_LinearCombinationKernel(
    ImageBuffer<InPixelT>& inImageImg,
    ImageBuffer<VarPixel>& inImageVar,
    ImageBuffer<MskPixel>& inImageMsk,
    std::vector<double> colPos,
    std::vector<double> rowPos,
    std::vector< lsst::afw::math::Kernel::SpatialFunctionPtr > sFn,
    ImageBuffer<OutPixelT>&                outImageImg,
    ImageBuffer<VarPixel>&                 outImageVar,
    ImageBuffer<MskPixel>&                 outImageMsk,
    std::vector< ImageBuffer<KerPixel> >&  basisKernels,
    SpatialFunctionType_t sfType,
    bool doNormalize
);

}
}
}
} //namespace lsst::afw::math::detail ends

#endif //GPU_BUILD

