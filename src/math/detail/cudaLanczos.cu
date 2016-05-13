// -*- LSST-C++ -*- // fixed format comment for emacs

/*
 * LSST Data Management System
 * Copyright 2008 - 2012 LSST Corporation.
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
 * \file
 *
 * \ingroup afw
 *
 * \brief GPU image warping CUDA implementation
 *
 * \author Kresimir Cosic.
 */

#define NVCC_COMPILING

#include <cstdint>

#include "lsst/afw/image/LsstImageTypes.h"
#include "lsst/afw/math/detail/CudaLanczos.h"


namespace lsst {
namespace afw {
namespace math {
namespace detail {
namespace gpu {


namespace
{
// CeilDivide: returns the smallest integer n such that n*divisor>=num
// preconditions: num>=0, divisor>0
__device__
int CeilDivide(int num, int divisor)    {
    return (num + divisor - 1) / divisor;
}

// Min function
template<typename T> __device__
T Min(T a, T b) {
    return a < b ? a : b;
}

// Max function
template<typename T> __device__
T Max(T a, T b) {
    return a > b ? a : b;
}

// Lanczos function
// precondition:  -order <= x <= order
template <typename T>
__device__ T Lanczos(T x, T orderInv)
{
    const T PI = 3.1415926535897932384626433832795028;
    const T xArg1 = fabs(x) * PI;
    if ( xArg1 > 1.0e-5) {
        const T xArg2 = xArg1 * orderInv;
        return sin(xArg1) * sin(xArg2) / (xArg1 * xArg2);
    }
    return T(1.0);
}

// Is Lanczos or bilinear function equal zero
__device__ bool IsEqualZeroLanczosOrBilinear(double x)
{
    if (x != floor(x)) return false;
    if (x == 0) return false;
    return true;
}

// Calculates the value of a single output pixel (for MaskedImage)
template<typename SrcPixelT>
__device__ PixelIVM<double> ApplyLanczosFilterMI(
    const ImageDataPtr<SrcPixelT> srcImage,
    int const srcX, int const srcY,
    int const mainKernelSize,
    const KernelType maskKernelType,
    int const maskKernelSize,
    double const kernelFracX, double const kernelFracY
)
{
    int const srcTLX = srcX + 1 - mainKernelSize / 2;
    int const srcTLY = srcY + 1 - mainKernelSize / 2;

    //calculate values of Lanczos function for rows
    double kernelRowVal[SIZE_MAX_WARPING_KERNEL];
    for (int kernelX = 0; kernelX < mainKernelSize; kernelX++) {
        kernelRowVal[kernelX] = Lanczos(1 - mainKernelSize / 2 - kernelFracX + kernelX, 2.0 / mainKernelSize);
    }

    double   colSumImg = 0;
    double   colSumVar = 0;
    MskPixel colSumMsk = 0;
    double kernelSum = 0;

    if (maskKernelType == KERNEL_TYPE_LANCZOS && mainKernelSize == maskKernelSize) {
        // mask kernel is identical to main kernel
        for (int kernelY = 0; kernelY < mainKernelSize; kernelY++) {
            double   rowSumImg = 0;
            double   rowSumVar = 0;
            MskPixel rowSumMsk = 0;
            double   rowKernelSum = 0;

            int srcPosImg = srcTLX + srcImage.strideImg * (srcTLY + kernelY);
            int srcPosVar = srcTLX + srcImage.strideVar * (srcTLY + kernelY);
            int srcPosMsk = srcTLX + srcImage.strideMsk * (srcTLY + kernelY);

            for (int kernelX = 0; kernelX < mainKernelSize; kernelX++) {
                double   srcImgPixel = srcImage.img[srcPosImg++];
                double   srcVarPixel = srcImage.var[srcPosVar++];
                MskPixel srcMskPixel = srcImage.msk[srcPosMsk++];
                double kernelVal = kernelRowVal[kernelX];
                if (kernelVal != 0) {
                    rowSumImg += srcImgPixel * kernelVal;
                    rowSumVar += srcVarPixel * kernelVal * kernelVal;
                    rowSumMsk |= srcMskPixel;
                    rowKernelSum += kernelVal;
                }
            }

            double kernelVal = Lanczos(1 - mainKernelSize / 2 - kernelFracY + kernelY, 2.0 / mainKernelSize);
            if (kernelVal != 0) {
                colSumImg += rowSumImg * kernelVal;
                colSumVar += rowSumVar * kernelVal * kernelVal;
                colSumMsk |= rowSumMsk;
                kernelSum += rowKernelSum * kernelVal;
            }
        }
    } else { // mask kernel not identical to main kernel

        // variance and image kernel
        for (int kernelY = 0; kernelY < mainKernelSize; kernelY++) {
            double   rowSumImg = 0;
            double   rowSumVar = 0;
            double   rowKernelSum = 0;

            int srcPosImg = srcTLX + srcImage.strideImg * (srcTLY + kernelY);
            int srcPosVar = srcTLX + srcImage.strideVar * (srcTLY + kernelY);

            for (int kernelX = 0; kernelX < mainKernelSize; kernelX++) {
                double   srcImgPixel = srcImage.img[srcPosImg++];
                double   srcVarPixel = srcImage.var[srcPosVar++];
                double kernelVal = kernelRowVal[kernelX];
                if (kernelVal != 0) {
                    rowSumImg += srcImgPixel * kernelVal;
                    rowSumVar += srcVarPixel * kernelVal * kernelVal;
                    rowKernelSum += kernelVal;
                }
            }

            double kernelVal = Lanczos(1 - mainKernelSize / 2 - kernelFracY + kernelY, 2.0 / mainKernelSize);
            if (kernelVal != 0) {
                colSumImg += rowSumImg * kernelVal;
                colSumVar += rowSumVar * kernelVal * kernelVal;
                kernelSum += rowKernelSum * kernelVal;
            }
        }

        if (maskKernelType == KERNEL_TYPE_NEAREST_NEIGHBOR) {
            int const srcTLXMask = srcX;
            int const srcTLYMask = srcY;

            int const kernelX = int(kernelFracX + 0.5);
            int const kernelY = int(kernelFracY + 0.5);

            int srcPosMsk = srcTLXMask + kernelX + srcImage.strideMsk * (srcTLYMask + kernelY);
            MskPixel srcMskPixel = srcImage.msk[srcPosMsk];
            colSumMsk = srcMskPixel;
        } else { // lanczos or bilinear mask kernel
            int const srcTLXMask = srcX + 1 - maskKernelSize / 2;
            int const srcTLYMask = srcY + 1 - maskKernelSize / 2;

            for (int kernelY = 0; kernelY < maskKernelSize; kernelY++) {
                if (IsEqualZeroLanczosOrBilinear(1 - maskKernelSize / 2 - kernelFracY + kernelY) ) continue;

                int srcPosMsk = srcTLXMask + srcImage.strideMsk * (srcTLYMask + kernelY);
                for (int kernelX = 0; kernelX < maskKernelSize; kernelX++, srcPosMsk++) {
                    if (!IsEqualZeroLanczosOrBilinear(1 - maskKernelSize / 2 - kernelFracX + kernelX)) {
                        MskPixel srcMskPixel = srcImage.msk[srcPosMsk];
                        colSumMsk |= srcMskPixel;
                    }
                }
            }
        }

    }

    PixelIVM<double> ret;
    ret.img = colSumImg / kernelSum;
    ret.var = colSumVar / (kernelSum * kernelSum);
    ret.msk = colSumMsk;
    return ret;
}

// Calculates the value of a single output pixel (for plain image)
template<typename SrcPixelT>
__device__ double ApplyLanczosFilter(const SrcPixelT* srcImgPtr, int const srcImgStride, int const srcWidth,
                                     int const srcX, int const srcY,
                                     int const mainKernelSize,
                                     double const kernelFracX, double const kernelFracY
                                    )
{
    int const srcTLX = srcX + 1 - mainKernelSize / 2;
    int const srcTLY = srcY + 1 - mainKernelSize / 2;

    //calculate values of Lanczos function for rows
    double kernelRowVal[SIZE_MAX_WARPING_KERNEL];
    for (int kernelX = 0; kernelX < mainKernelSize; kernelX++) {
        kernelRowVal[kernelX] = Lanczos(1 - mainKernelSize / 2 - kernelFracX + kernelX, 2.0 / mainKernelSize);
    }

    double colSumImg = 0;
    double kernelSum = 0;
    for (int kernelY = 0; kernelY < mainKernelSize; kernelY++) {
        double rowSumImg = 0;
        double rowKernelSum = 0;
        int srcPosImg = srcTLX + srcImgStride * (srcTLY + kernelY);
        for (int kernelX = 0; kernelX < mainKernelSize; kernelX++) {
            double   srcImgPixel = srcImgPtr[srcPosImg++];
            double kernelVal = kernelRowVal[kernelX];
            if (kernelVal != 0) {
                rowSumImg += srcImgPixel * kernelVal;
                rowKernelSum += kernelVal;
            }
        }

        double kernelVal = Lanczos(1 - mainKernelSize / 2 - kernelFracY + kernelY, 2.0 / mainKernelSize);
        if (kernelVal != 0) {
            colSumImg += rowSumImg * kernelVal;
            kernelSum += rowKernelSum * kernelVal;
        }
    }

    return colSumImg / kernelSum;
}

// calculate the interpolated value given the data for bilinear interpolation
__device__ SPoint2 GetInterpolatedValue(BilinearInterp* const interpBuf, int interpBufPitch,
                                        int interpLen, int x, int y
                                       )
{
    int blkX = x / interpLen;
    int blkY = y / interpLen;

    int subX = x - blkX * interpLen;
    int subY = y - blkY * interpLen;

    BilinearInterp interp = interpBuf[blkX + blkY* interpBufPitch];
    return interp.Interpolate(subX, subY);
}

} //local namespace ends

/// GPU kernel for lanczos resampling
template<typename DestPixelT, typename SrcPixelT>
__global__ void WarpImageGpuKernel(
    bool isMaskedImage,
    ImageDataPtr<DestPixelT> destImage,
    ImageDataPtr<SrcPixelT>  srcImage,
    int const mainKernelSize,
    const KernelType maskKernelType,
    int const maskKernelSize,
    SBox2I srcGoodBox,
    PixelIVM<DestPixelT> edgePixel,
    BilinearInterp* srcPosInterp,
    int interpLength
)
{
    int const blockSizeX = SIZE_X_WARPING_BLOCK;
    int const blockSizeY = SIZE_Y_WARPING_BLOCK;

    //number of blocks in X and Y directions
    int const blockNX = CeilDivide(destImage.width,  blockSizeX);
    int const blockNY = CeilDivide(destImage.height, blockSizeY);

    int const totalBlocks = blockNX * blockNY;

    // calculates pitch of srcPosInterp array
    int const srcPosInterpPitch = CeilDivide(destImage.width, interpLength) + 1;

    // for each block of destination image
    for (int blkI = blockIdx.x; blkI < totalBlocks; blkI += gridDim.x)
    {
        // claculate coordinates of the block that is being processed
        int const blkIX = blkI % blockNX;
        int const blkIY = blkI / blockNX;

        // coordinate of upper left corner of the block
        int const blkX = blkIX * blockSizeX;
        int const blkY = blkIY * blockSizeY;

        // Each thread gets its own pixel.
        // The calling function ensures that the number of pixels in a block
        // matches the number of threads in a block
        // (or less pixels than threads for blocks on the edge)
        int const curBlkPixelX = threadIdx.x % blockSizeX;
        int const curBlkPixelY = threadIdx.x / blockSizeX;

        // calculate the position of a destination pixel for current thread
        int const pixelX = blkX + curBlkPixelX;
        int const pixelY = blkY + curBlkPixelY;

        // On edges: skip calculation for threads that got pixels which are outside the destination image
        if (pixelX >= destImage.width || pixelY >= destImage.height) continue;

        // srcPos - position in source (of Lanczos kernel center)
        // calculated as a linear interpolation of the transformation function
        const SPoint2 srcPos = GetInterpolatedValue(srcPosInterp, srcPosInterpPitch,
                               interpLength, pixelX + 1, pixelY + 1);
        double const roundedSrcPtX = floor(srcPos.x);
        double const roundedSrcPtY = floor(srcPos.y);
        //integer and frac parts of the kernel center
        int const    srcX = int(roundedSrcPtX);
        int const    srcY = int(roundedSrcPtY);
        double const kernelFracX = srcPos.x - roundedSrcPtX;
        double const kernelFracY = srcPos.y - roundedSrcPtY;

        // check that destination pixel is mapped from within the source image
        if (   srcGoodBox.begX <= srcX && srcX < srcGoodBox.endX
                && srcGoodBox.begY <= srcY && srcY < srcGoodBox.endY
           ) {
            //relative area
            const SPoint2 leftSrcPos = GetInterpolatedValue(srcPosInterp, srcPosInterpPitch,
                                       interpLength, pixelX, pixelY + 1);
            const SPoint2 upSrcPos = GetInterpolatedValue(srcPosInterp, srcPosInterpPitch,
                                     interpLength, pixelX + 1, pixelY);

            const SVec2 dSrcA = SVec2(leftSrcPos, srcPos);
            const SVec2 dSrcB = SVec2(upSrcPos, srcPos);
            double const relativeArea = fabs(dSrcA.x * dSrcB.y - dSrcA.y * dSrcB.x);

            if (isMaskedImage) {
                const PixelIVM<double> sample = ApplyLanczosFilterMI(srcImage,
                                                srcX, srcY,
                                                mainKernelSize, maskKernelType, maskKernelSize,
                                                kernelFracX, kernelFracY
                                                                    );
                int const pixelIimg = pixelY * destImage.strideImg + pixelX;
                int const pixelIvar = pixelY * destImage.strideVar + pixelX;
                int const pixelImsk = pixelY * destImage.strideMsk + pixelX;

                destImage.img[pixelIimg] = sample.img * relativeArea;
                destImage.var[pixelIvar] = sample.var * relativeArea * relativeArea;
                destImage.msk[pixelImsk] = sample.msk;
            } else {
                double sample = ApplyLanczosFilter(srcImage.img, srcImage.strideImg, srcImage.width,
                                                   srcX, srcY,
                                                   mainKernelSize, kernelFracX, kernelFracY
                                                  );
                int const pixelIimg = pixelY * destImage.strideImg + pixelX; //pixel index in destination image
                destImage.img[pixelIimg] = sample * relativeArea;
            }
        } else {
            //set the output pixel to the value of edgePixel
            int const pixelIimg = pixelY * destImage.strideImg + pixelX; //pixel index in destination image
            destImage.img[pixelIimg] = edgePixel.img;
            if (isMaskedImage) {
                int const pixelIvar = pixelY * destImage.strideVar + pixelX;
                int const pixelImsk = pixelY * destImage.strideMsk + pixelX;
                destImage.var[pixelIvar] = edgePixel.var;
                destImage.msk[pixelImsk] = edgePixel.msk;
            }
        }
    }
}

// External interface, calls the GPU kernel for lanczos resampling
template<typename DestPixelT, typename SrcPixelT>
void WarpImageGpuCallKernel(bool isMaskedImage,
                            ImageDataPtr<DestPixelT> destImageGpu,
                            ImageDataPtr<SrcPixelT>  srcImageGpu,
                            int mainKernelSize,
                            KernelType maskKernelType,
                            int maskKernelSize,
                            SBox2I srcGoodBox,
                            PixelIVM<DestPixelT> edgePixel,
                            BilinearInterp* srcPosInterp,
                            int interpLength
                           )
{
    dim3 block(SIZE_X_WARPING_BLOCK * SIZE_Y_WARPING_BLOCK);
    dim3 grid(7 * 16); //divisible by no. of SM's in most GPUs, performs well

    WarpImageGpuKernel <<< grid, block, 0>>>(
        isMaskedImage,
        destImageGpu,
        srcImageGpu,
        mainKernelSize,
        maskKernelType,
        maskKernelSize,
        srcGoodBox,
        edgePixel,
        srcPosInterp,
        interpLength
    );
}

//
// Explicit instantiations
//
/// \cond
#define MASKEDIMAGE(PIXTYPE) afwImage::MaskedImage<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
#define IMAGE(PIXTYPE) afwImage::Image<PIXTYPE>
#define NL /* */

#define INSTANTIATE(DESTIMAGEPIXELT, SRCIMAGEPIXELT) \
    template void WarpImageGpuCallKernel(  \
                            bool isMaskedImage, \
                            ImageDataPtr<DESTIMAGEPIXELT> destImageGpu, \
                            ImageDataPtr<SRCIMAGEPIXELT>  srcImageGpu, \
                            int mainKernelSize, \
                            KernelType maskKernelType, \
                            int maskKernelSize, \
                            SBox2I srcGoodBox, \
                            PixelIVM<DESTIMAGEPIXELT> edgePixel, \
                            BilinearInterp* srcPosInterp, \
                            int interpLength \
                            );

INSTANTIATE(double, double)
INSTANTIATE(double, float)
INSTANTIATE(double, int)
INSTANTIATE(double, std::uint16_t)
INSTANTIATE(float, float)
INSTANTIATE(float, int)
INSTANTIATE(float, std::uint16_t)
INSTANTIATE(int, int)
INSTANTIATE(std::uint16_t, std::uint16_t)
/// \endcond

}
}
}
}
} //namespace lsst::afw::math::detail::gpu ends

