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

#include "lsst/afw/image/LsstImageTypes.h"
#include "lsst/afw/math/detail/CudaLanczos.h"


namespace lsst {
namespace afw {
namespace math {
namespace detail {
namespace gpu {

typedef unsigned char uint8;
extern __shared__ uint8 smem[];

namespace
{
// CeilDivide: returns the greatest integer n such that n*divisor>=num
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


// Calculates the value of a single output pixel (for MaskedImage)
template<typename SrcPixelT>
__device__ PixelIVM<double> ApplyLanczosFilterMI(int kernelSize, double orderInv,
        ImageDataPtr<SrcPixelT> srcImage,
        int srcX, int srcY,
        int kernelCenterX, int kernelCenterY,
        double kernelFracX, double kernelFracY
                                                )
{
    const int srcTLX = srcX - kernelCenterX;
    const int srcTLY = srcY - kernelCenterY;

    //calculate values of Lanczos function for rows
    double kernelRowVal[cWarpingKernelMaxSize];
    for (int kernelX = 0; kernelX < kernelSize; kernelX++) {
        kernelRowVal[kernelX] = Lanczos(-kernelCenterX - kernelFracX + kernelX, orderInv);
    }

    double   colSumImg = 0;
    double   colSumVar = 0;
    MskPixel colSumMsk = 0;

    double kernelSum = 0;
    for (int kernelY = 0; kernelY < kernelSize; kernelY++) {
        double rowSumImg = 0;
        double rowSumVar = 0;
        double rowKernelSum = 0;

        int srcPosImg = srcTLX + srcImage.strideImg * (srcTLY + kernelY);
        int srcPosVar = srcTLX + srcImage.strideVar * (srcTLY + kernelY);
        int srcPosMsk = srcTLX + srcImage.strideMsk * (srcTLY + kernelY);

        for (int kernelX = 0; kernelX < kernelSize; kernelX++) {
            double   srcImgPixel = srcImage.img[srcPosImg++];
            double   srcVarPixel = srcImage.var[srcPosVar++];
            MskPixel srcMskPixel = srcImage.msk[srcPosMsk++];
            double kernelVal = kernelRowVal[kernelX];
            if (kernelVal != 0) {
                rowSumImg += srcImgPixel * kernelVal;
                rowSumVar += srcVarPixel * kernelVal * kernelVal;
                colSumMsk |= srcMskPixel;
                rowKernelSum += kernelVal;
            }
        }

        double kernelVal = Lanczos(-kernelCenterY - kernelFracY + kernelY, orderInv);
        if (kernelVal != 0) {
            colSumImg += rowSumImg * kernelVal;
            colSumVar += rowSumVar * kernelVal * kernelVal;
            kernelSum += rowKernelSum * kernelVal;
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
__device__ double ApplyLanczosFilter(int kernelSize, double orderInv,
                                     SrcPixelT* srcImgPtr, int srcImgStride, int srcWidth,
                                     int srcX, int srcY,
                                     int kernelCenterX, int kernelCenterY,
                                     double kernelFracX, double kernelFracY
                                    )
{
    const int srcTLX = srcX - kernelCenterX;
    const int srcTLY = srcY - kernelCenterY;

    //calculate values of Lanczos function for rows
    double kernelRowVal[cWarpingKernelMaxSize];
    for (int kernelX = 0; kernelX < kernelSize; kernelX++) {
        kernelRowVal[kernelX] = Lanczos(-kernelCenterX - kernelFracX + kernelX, orderInv);
    }

    double colSumImg = 0;
    double kernelSum = 0;
    for (int kernelY = 0; kernelY < kernelSize; kernelY++) {
        double rowSumImg = 0;
        double rowKernelSum = 0;
        int srcPosImg = srcTLX + srcImgStride * (srcTLY + kernelY);
        for (int kernelX = 0; kernelX < kernelSize; kernelX++) {
            double   srcImgPixel = srcImgPtr[srcPosImg++];
            double kernelVal = kernelRowVal[kernelX];
            if (kernelVal != 0) {
                rowSumImg += srcImgPixel * kernelVal;
                rowKernelSum += kernelVal;
            }
        }

        double kernelVal = Lanczos(-kernelCenterY - kernelFracY + kernelY, orderInv);
        if (kernelVal != 0) {
            colSumImg += rowSumImg * kernelVal;
            kernelSum += rowKernelSum * kernelVal;
        }
    }

    return colSumImg / kernelSum;
}

// calculate the interpolated value given the data for linear interpolation
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
    int order,
    SBox2I srcGoodBox,
    int kernelCenterX,
    int kernelCenterY,
    PixelIVM<DestPixelT> edgePixel,
    SBox2I* srcBlk,
    BilinearInterp* srcPosInterp,
    int interpLength
)
{
    const double orderInv = 1.0 / order;
    const int kernelSize = order * 2;

    const int blockSizeX = cWarpingBlockSizeX;
    const int blockSizeY = cWarpingBlockSizeY;

    //number of blocks in X nad Y directions
    const int blockNX = CeilDivide(destImage.width,  blockSizeX);
    const int blockNY = CeilDivide(destImage.height, blockSizeY);

    const int totalBlocks = blockNX * blockNY;

    // calculates pitch of srcPosInterp array
    const int srcPosInterpPitch = CeilDivide(destImage.width, interpLength) + 1;

    // for each block of destination image
    for (int blkI = blockIdx.x; blkI < totalBlocks; blkI += gridDim.x)
    {
        // claculate coordinates of the block that is being processed
        const int blkIX = blkI % blockNX;
        const int blkIY = blkI / blockNX;

        // coordinate of upper left corner of the block
        const int blkX = blkIX * blockSizeX;
        const int blkY = blkIY * blockSizeY;

        // Each thread gets its own pixel.
        // The calling function ensures that the number of pixels in a block
        // matches the number of threads in a block
        // (or less pixels than threads for blocks on the edge)
        const int curBlkPixelX = threadIdx.x % blockSizeX;
        const int curBlkPixelY = threadIdx.x / blockSizeX;

        // calculate the position of a destination pixel for current thread
        const int pixelX = blkX + curBlkPixelX;
        const int pixelY = blkY + curBlkPixelY;

        // On edges: skip calculation for threads that got pixels which are outside the destination image
        if (pixelX >= destImage.width || pixelY >= destImage.height) continue;

        // srcPos - position in source (of Lanczos kernel center)
        // calculated as a linear interpolation of the transformation function
        const SPoint2 srcPos = GetInterpolatedValue(srcPosInterp, srcPosInterpPitch,
                               interpLength, pixelX + 1, pixelY + 1);
        const double roundedSrcPtX = floor(srcPos.x);
        const double roundedSrcPtY = floor(srcPos.y);
        //integer and frac parts of the kernel center
        const int    srcX = int(roundedSrcPtX);
        const int    srcY = int(roundedSrcPtY);
        const double kernelFracX = srcPos.x - roundedSrcPtX;
        const double kernelFracY = srcPos.y - roundedSrcPtY;

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
            const double relativeArea = fabs(dSrcA.x * dSrcB.y - dSrcA.y * dSrcB.x);

            if (isMaskedImage) {
                const PixelIVM<double> sample = ApplyLanczosFilterMI(kernelSize, orderInv, srcImage,
                                                srcX, srcY,
                                                kernelCenterX, kernelCenterY, kernelFracX, kernelFracY
                                                                    );
                const int pixelIimg = pixelY * destImage.strideImg + pixelX;
                const int pixelIvar = pixelY * destImage.strideVar + pixelX;
                const int pixelImsk = pixelY * destImage.strideMsk + pixelX;

                destImage.img[pixelIimg] = sample.img * relativeArea;
                destImage.var[pixelIvar] = sample.var * relativeArea * relativeArea;
                destImage.msk[pixelImsk] = sample.msk;
            } else {
                double sample = ApplyLanczosFilter(kernelSize, orderInv,
                                                   srcImage.img, srcImage.strideImg, srcImage.width,
                                                   srcX, srcY,
                                                   kernelCenterX, kernelCenterY, kernelFracX, kernelFracY
                                                  );
                const int pixelIimg = pixelY * destImage.strideImg + pixelX; //pixel index in destination image
                destImage.img[pixelIimg] = sample * relativeArea;
            }
        } else {
            //set the output pixel to the value of edgePixel
            const int pixelIimg = pixelY * destImage.strideImg + pixelX; //pixel index in destination image
            destImage.img[pixelIimg] = edgePixel.img;
            if (isMaskedImage) {
                const int pixelIvar = pixelY * destImage.strideVar + pixelX;
                const int pixelImsk = pixelY * destImage.strideMsk + pixelX;
                destImage.var[pixelIvar] = edgePixel.var;
                destImage.msk[pixelImsk] = edgePixel.msk;
            }
        }
    }
}

// In public interface, Calls the GPU kernel for lanczos resampling
template<typename DestPixelT, typename SrcPixelT>
void WarpImageGpuCallKernel(bool isMaskedImage,
                            ImageDataPtr<DestPixelT> destImageGpu,
                            ImageDataPtr<SrcPixelT>  srcImageGpu,
                            int order,
                            SBox2I srcGoodBox,
                            int kernelCenterX,
                            int kernelCenterY,
                            PixelIVM<DestPixelT> edgePixel,
                            SBox2I* srcBlk,
                            BilinearInterp* srcPosInterp,
                            int interpLength
                           )
{
    dim3 block(cWarpingBlockSizeX * cWarpingBlockSizeY);
    dim3 grid(14 * 8);

    WarpImageGpuKernel <<< grid, block, 15500>>>(
        isMaskedImage,
        destImageGpu,
        srcImageGpu,
        order,
        srcGoodBox,
        kernelCenterX,
        kernelCenterY,
        edgePixel,
        srcBlk,
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
                            int order, \
                            SBox2I srcGoodBox, \
                            int kernelCenterX, \
                            int kernelCenterY, \
                            PixelIVM<DESTIMAGEPIXELT> edgePixel, \
                            SBox2I* srcBlk, \
                            BilinearInterp* srcPosInterp, \
                            int interpLength \
                            );

INSTANTIATE(double, double)
INSTANTIATE(double, float)
INSTANTIATE(double, int)
INSTANTIATE(double, boost::uint16_t)
INSTANTIATE(float, float)
INSTANTIATE(float, int)
INSTANTIATE(float, boost::uint16_t)
INSTANTIATE(int, int)
INSTANTIATE(boost::uint16_t, boost::uint16_t)
/// \endcond

}
}
}
}
} //namespace lsst::afw::math::detail::gpu ends

