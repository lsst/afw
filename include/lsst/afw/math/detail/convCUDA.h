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

// image block size per GPU block. (The size of the image that one GPU block processes)
#define blockSizeX 32
#define blockSizeY 16

template <typename T>
void CallTestGpuKernel(T* ret);

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

}}}}}  //namespace lsst::afw::math::detail::gpu ends

