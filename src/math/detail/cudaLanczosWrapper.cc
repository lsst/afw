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
 * \brief GPU image warping implementation
 *
 * \author Kresimir Cosic.
 */

#ifdef GPU_BUILD
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "lsst/afw/geom/Box.h"

#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/warpExposure.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/image/Wcs.h"

#include "lsst/afw/gpu/GpuExceptions.h"
#include "lsst/afw/gpu/IsGpuBuild.h"
#include "lsst/afw/gpu/detail/GpuBuffer2D.h"
#include "lsst/afw/gpu/detail/CudaMemory.h"
#include "lsst/afw/gpu/detail/CudaSelectGpu.h"

#include "lsst/afw/math/detail/CudaLanczos.h"
#include "lsst/afw/math/detail/CudaLanczosWrapper.h"

using namespace std;
using lsst::afw::math::detail::gpu::SPoint2;
using lsst::afw::math::detail::gpu::SVec2;
using lsst::afw::math::detail::gpu::SBox2I;


namespace mathDetail = lsst::afw::math::detail;
namespace gpuDetail = lsst::afw::gpu::detail;
namespace afwMath = lsst::afw::math;
namespace afwGpu = lsst::afw::gpu;
namespace afwImage = lsst::afw::image;
namespace pexExcept = lsst::pex::exceptions;
namespace pexLog = lsst::pex::logging;
namespace afwGeom = lsst::afw::geom;

namespace lsst {
namespace afw {
namespace math {
namespace detail {

namespace {

int CeilDivide(int num, int divisor)
{
    return (num + divisor - 1) / divisor;
}

// get the number of interpolation blocks given an image dimension
int InterpBlkN(int size , int interpLength)
{
    return CeilDivide(size , interpLength) + 1;
}

// calculate the interpolated value given the data for linear interpolation
gpu::SPoint2 GetInterpolatedValue(afwGpu::detail::GpuBuffer2D<gpu::BilinearInterp> const & interpBuf,
                                  int blkX, int blkY, int subX, int subY
                                 )
{
    gpu::BilinearInterp interp = interpBuf.Pixel(blkX, blkY);
    return interp.Interpolate(subX, subY);
}

// calculate the interpolated value given the data for linear interpolation
gpu::SPoint2 GetInterpolatedValue(afwGpu::detail::GpuBuffer2D<gpu::BilinearInterp> const & interpBuf,
                                  int interpLen, int x, int y
                                 )
{
    int blkX = x / interpLen;
    int blkY = y / interpLen;

    int subX = x % interpLen;
    int subY = y % interpLen;

    return GetInterpolatedValue(interpBuf, blkX, blkY, subX, subY);
}

// calculate the number of points falling within the srcGoodBox,
// given a bilinearily interpolated coordinate transform function on integer range [0,width> x [0, height>
int NumGoodPixels(afwGpu::detail::GpuBuffer2D<gpu::BilinearInterp> const & interpBuf,
                  int const interpLen, int const width, int const height, SBox2I srcGoodBox)
{
    int cnt = 0;

    int subY = 1, blkY = 0;
    for (int row = 0; row < height; row++, subY++) {
        if (subY >= interpLen) {
            subY -= interpLen;
            blkY++;
        }

        int subX = 1, blkX = 0;
        gpu::BilinearInterp interp = interpBuf.Pixel(blkX, blkY);
        gpu::LinearInterp lineY = interp.GetLinearInterp(subY);

        for (int col = 0; col < width; col++, subX++) {
            if (subX >= interpLen) {
                subX -= interpLen;
                blkX++;
                interp = interpBuf.Pixel(blkX, blkY);
                lineY = interp.GetLinearInterp(subY);
            }
            gpu::SPoint2 srcPos = lineY.Interpolate(subX);
            if (srcGoodBox.isInsideBox(srcPos)) {
                cnt++;
            }
        }
    }
    return cnt;
}

#ifdef GPU_BUILD
// for (plain) Image::
// allocate CPU and GPU buffers, transfer data and call GPU kernel proxy
// precondition: order*2 < gpu::SIZE_MAX_WARPING_KERNEL
template< typename DestPixelT, typename SrcPixelT>
int WarpImageGpuWrapper(
    afwImage::Image<DestPixelT>     &destImage,
    afwImage::Image<SrcPixelT> const &srcImage,
    int mainKernelSize,
    gpu::KernelType maskKernelType,
    int maskKernelSize,
    const lsst::afw::geom::Box2I srcBox,
    lsst::afw::gpu::detail::GpuBuffer2D<gpu::BilinearInterp> const& srcPosInterp,
    int const interpLength,
    typename afwImage::Image<DestPixelT>::SinglePixel padValue
)
{
    typedef typename afwImage::Image<DestPixelT> DestImageT;

    typename DestImageT::SinglePixel const edgePixel = padValue;

    gpu::PixelIVM<DestPixelT> edgePixelGpu;
    edgePixelGpu.img = edgePixel;
    edgePixelGpu.var = -1;
    edgePixelGpu.msk = -1;

    int const destWidth = destImage.getWidth();
    int const destHeight = destImage.getHeight();
    gpuDetail::GpuMemOwner<DestPixelT> destBufImgGpu;
    gpuDetail::GpuMemOwner<SrcPixelT> srcBufImgGpu;
    gpuDetail::GpuMemOwner<gpu::BilinearInterp> srcPosInterpGpu;

    gpu::ImageDataPtr<DestPixelT> destImgGpu;
    destImgGpu.strideImg = destBufImgGpu.AllocImageBaseBuffer(destImage);
    if (destBufImgGpu.ptr == NULL)  {
        throw LSST_EXCEPT(afwGpu::GpuMemoryError, "Not enough memory on GPU for output image");
    }
    destImgGpu.img = destBufImgGpu.ptr;
    destImgGpu.var = NULL;
    destImgGpu.msk = NULL;
    destImgGpu.width = destWidth;
    destImgGpu.height = destHeight;

    gpu::ImageDataPtr<SrcPixelT> srcImgGpu;
    srcImgGpu.strideImg = srcBufImgGpu.TransferFromImageBase(srcImage);
    if (srcBufImgGpu.ptr == NULL)  {
        throw LSST_EXCEPT(afwGpu::GpuMemoryError, "Not enough memory on GPU for input image");
    }
    srcImgGpu.img = srcBufImgGpu.ptr;
    srcImgGpu.var = NULL;
    srcImgGpu.msk = NULL;
    srcImgGpu.width = srcImage.getWidth();
    srcImgGpu.height = srcImage.getHeight();

    srcPosInterpGpu.Transfer(srcPosInterp);
    if (srcPosInterpGpu.ptr == NULL)  {
        throw LSST_EXCEPT(afwGpu::GpuMemoryError,
                          "Not enough memory on GPU for interpolation data for coorinate transformation");
    }

    SBox2I srcBoxConv(srcBox.getMinX(), srcBox.getMinY(), srcBox.getMaxX() + 1, srcBox.getMaxY() + 1);

    WarpImageGpuCallKernel(false,
                           destImgGpu, srcImgGpu,
                           mainKernelSize,
                           maskKernelType,
                           maskKernelSize,
                           srcBoxConv,
                           edgePixelGpu,
                           srcPosInterpGpu.ptr, interpLength
                          );

    int numGoodPixels = NumGoodPixels(srcPosInterp, interpLength, destWidth, destHeight, srcBoxConv);

    cudaThreadSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        throw LSST_EXCEPT(afwGpu::GpuRuntimeError, "GPU calculation failed to run");
    }

    destBufImgGpu.CopyToImageBase(destImage);
    return numGoodPixels;
}

// for MaskedImage::
// allocate CPU and GPU buffers, transfer data and call GPU kernel proxy
// precondition: order*2 < gpu::SIZE_MAX_WARPING_KERNEL
template< typename DestPixelT, typename SrcPixelT>
int WarpImageGpuWrapper(
    afwImage::MaskedImage<DestPixelT>      &dstImage,
    afwImage::MaskedImage<SrcPixelT>const  &srcImage,
    int mainKernelSize,
    gpu::KernelType maskKernelType,
    int maskKernelSize,
    const lsst::afw::geom::Box2I srcBox,
    lsst::afw::gpu::detail::GpuBuffer2D<gpu::BilinearInterp> const& srcPosInterp,
    int const interpLength,
    typename afwImage::MaskedImage<DestPixelT>::SinglePixel padValue
)
{
    typedef typename afwImage::MaskedImage<DestPixelT> DestImageT;

    typename DestImageT::SinglePixel const edgePixel = padValue;

    gpu::PixelIVM<DestPixelT> edgePixelGpu;
    edgePixelGpu.img = edgePixel.image();
    edgePixelGpu.var = edgePixel.variance();
    edgePixelGpu.msk = edgePixel.mask();

    int const destWidth = dstImage.getWidth();
    int const destHeight = dstImage.getHeight();

    gpuDetail::GpuMemOwner<DestPixelT> destBufImgGpu;
    gpuDetail::GpuMemOwner<gpu::VarPixel>   destBufVarGpu;
    gpuDetail::GpuMemOwner<gpu::MskPixel>   destBufMskGpu;

    gpuDetail::GpuMemOwner<SrcPixelT> srcBufImgGpu;
    gpuDetail::GpuMemOwner<gpu::VarPixel>  srcBufVarGpu;
    gpuDetail::GpuMemOwner<gpu::MskPixel>  srcBufMskGpu;

    gpuDetail::GpuMemOwner<gpu::BilinearInterp> srcPosInterpGpu;

    mathDetail::gpu::ImageDataPtr<DestPixelT> destImgGpu;
    destImgGpu.strideImg = destBufImgGpu.AllocImageBaseBuffer(*dstImage.getImage());
    destImgGpu.strideVar = destBufVarGpu.AllocImageBaseBuffer(*dstImage.getVariance());
    destImgGpu.strideMsk = destBufMskGpu.AllocImageBaseBuffer(*dstImage.getMask());
    if (destBufImgGpu.ptr == NULL)  {
        throw LSST_EXCEPT(afwGpu::GpuMemoryError, "Not enough memory on GPU for output image");
    }
    if (destBufVarGpu.ptr == NULL)  {
        throw LSST_EXCEPT(afwGpu::GpuMemoryError, "Not enough memory on GPU for output variance");
    }
    if (destBufMskGpu.ptr == NULL)  {
        throw LSST_EXCEPT(afwGpu::GpuMemoryError, "Not enough memory on GPU for output mask");
    }
    destImgGpu.img = destBufImgGpu.ptr;
    destImgGpu.var = destBufVarGpu.ptr;
    destImgGpu.msk = destBufMskGpu.ptr;
    destImgGpu.width = destWidth;
    destImgGpu.height = destHeight;

    gpu::ImageDataPtr<SrcPixelT> srcImgGpu;
    srcImgGpu.strideImg = srcBufImgGpu.TransferFromImageBase(*srcImage.getImage());
    if (srcBufImgGpu.ptr == NULL)  {
        throw LSST_EXCEPT(afwGpu::GpuMemoryError, "Not enough memory on GPU for input image");
    }
    srcImgGpu.strideVar = srcBufVarGpu.TransferFromImageBase(*srcImage.getVariance());
    if (srcBufVarGpu.ptr == NULL)  {
        throw LSST_EXCEPT(afwGpu::GpuMemoryError, "Not enough memory on GPU for input variance");
    }
    srcImgGpu.strideMsk = srcBufMskGpu.TransferFromImageBase(*srcImage.getMask());
    if (srcBufMskGpu.ptr == NULL)  {
        throw LSST_EXCEPT(afwGpu::GpuMemoryError, "Not enough memory on GPU for input mask");
    }

    srcImgGpu.img = srcBufImgGpu.ptr;
    srcImgGpu.var = srcBufVarGpu.ptr;
    srcImgGpu.msk = srcBufMskGpu.ptr;
    srcImgGpu.width = srcImage.getWidth();
    srcImgGpu.height = srcImage.getHeight();

    srcPosInterpGpu.Transfer(srcPosInterp);
    if (srcPosInterpGpu.ptr == NULL)  {
        throw LSST_EXCEPT(afwGpu::GpuMemoryError,
                          "Not enough memory on GPU for interpolation data for coorinate transformation");
    }

    SBox2I srcBoxConv(srcBox.getMinX(), srcBox.getMinY(), srcBox.getMaxX() + 1, srcBox.getMaxY() + 1);

    WarpImageGpuCallKernel(true,
                           destImgGpu, srcImgGpu,
                           mainKernelSize,
                           maskKernelType,
                           maskKernelSize,
                           srcBoxConv,
                           edgePixelGpu,
                           srcPosInterpGpu.ptr, interpLength
                          );
    int numGoodPixels = NumGoodPixels(srcPosInterp, interpLength, destWidth, destHeight, srcBoxConv);

    cudaThreadSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        throw LSST_EXCEPT(afwGpu::GpuRuntimeError, "GPU calculation failed to run");
    }

    destBufImgGpu.CopyToImageBase(*dstImage.getImage());
    destBufVarGpu.CopyToImageBase(*dstImage.getVariance());
    destBufMskGpu.CopyToImageBase(*dstImage.getMask());

    return numGoodPixels;
}
#endif //GPU_BUILD

// Calculate bilinear interpolation data based on given function values
// input:
//    srcPosInterp - contains values of original function at a mesh of equally distanced points
//                  the values are stored in .o member
//    interpLength - distance between points
//    destWidth, destHeight - size of function domain
// output:
//    srcPosInterp - all members are calculated and set, ready to calculate interpolation values
void CalculateInterpolationData(gpuDetail::GpuBuffer2D<gpu::BilinearInterp>& srcPosInterp, int interpLength,
                                int destWidth, int destHeight)
{
    int const interpBlkNX = InterpBlkN(destWidth , interpLength);
    int const interpBlkNY = InterpBlkN(destHeight, interpLength);

    for (int row = -1, rowBand = 0; rowBand < interpBlkNY - 1; row += interpLength, rowBand++) {
        double const invInterpLen = 1.0 / interpLength;
        double const invInterpLenRow = row + interpLength <= destHeight - 1 ?
                                       invInterpLen : 1.0 / (destHeight - 1 - row);

        for (int col = -1, colBand = 0; colBand < interpBlkNX - 1; col += interpLength, colBand++) {

            const SPoint2 p11 = srcPosInterp.Pixel(colBand  , rowBand  ).o;
            const SPoint2 p12 = srcPosInterp.Pixel(colBand + 1, rowBand  ).o;
            const SPoint2 p21 = srcPosInterp.Pixel(colBand  , rowBand + 1).o;
            const SPoint2 p22 = srcPosInterp.Pixel(colBand + 1, rowBand + 1).o;
            const SVec2 band_dY  = SVec2(p11, p21);
            const SVec2 band_d0X = SVec2(p11, p12);
            const SVec2 band_d1X = SVec2(p21, p22);
            const SVec2 band_ddX = VecMul( VecSub(band_d1X, band_d0X), invInterpLenRow);

            double const invInterpLenCol = col + interpLength <= destWidth - 1 ?
                                           invInterpLen : 1.0 / (destWidth - 1 - col);

            gpu::BilinearInterp lin = srcPosInterp.Pixel(colBand, rowBand); //sets lin.o
            lin.deltaY = VecMul(band_dY , invInterpLenRow);
            lin.d0X    = VecMul(band_d0X, invInterpLenCol);
            lin.ddX    = VecMul(band_ddX, invInterpLenCol);
            srcPosInterp.Pixel(colBand, rowBand) = lin;

            // partially fill the last column and row, too
            if (colBand == interpBlkNX - 2) {
                srcPosInterp.Pixel(interpBlkNX - 1, rowBand).deltaY =
                    VecMul( SVec2(p12, p22), invInterpLenRow);
            }
            if (rowBand == interpBlkNY - 2) {
                srcPosInterp.Pixel(colBand, interpBlkNY - 1).d0X =
                    VecMul( SVec2(p21, p22), invInterpLenCol);
            }
        }
    }
}

} //local namespace ends

// a part of public interface, see header file for description
template<typename DestImageT, typename SrcImageT>
std::pair<int, WarpImageGpuStatus::ReturnCode> warpImageGPU(
    DestImageT &destImage,              ///< remapped %image
    SrcImageT const &srcImage,          ///< source %image
    afwMath::LanczosWarpingKernel const &lanczosKernel,     ///< warping kernel
    lsst::afw::math::SeparableKernel const &maskWarpingKernel,    ///< mask warping kernel
    PositionFunctor const &computeSrcPos,      ///< Functor to compute source position
    int const interpLength,              ///< Distance over which WCS can be linearily interpolated, must be >0
    typename DestImageT::SinglePixel padValue, ///< value to use for undefined pixels
    const bool forceProcessing
)
{
    if (interpLength < 1) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                          "GPU accelerated warping must use interpolation");
    }

    int const srcWidth = srcImage.getWidth();
    int const srcHeight = srcImage.getHeight();
    pexLog::TTrace<3>("lsst.afw.math.warp", "(GPU) source image width=%d; height=%d", srcWidth, srcHeight);

    if (!lsst::afw::gpu::isGpuBuild()) {
        throw LSST_EXCEPT(afwGpu::GpuRuntimeError, "Afw not compiled with GPU support");
    }

#ifdef GPU_BUILD
    gpu::KernelType maskKernelType;
    {
        if (dynamic_cast<afwMath::LanczosWarpingKernel const*>(&maskWarpingKernel)) {
            maskKernelType = gpu::KERNEL_TYPE_LANCZOS;
        } else if (dynamic_cast<afwMath::BilinearWarpingKernel const*>(&maskWarpingKernel)) {
            maskKernelType = gpu::KERNEL_TYPE_BILINEAR;
        } else if (dynamic_cast<afwMath::NearestWarpingKernel const*>(&maskWarpingKernel)) {
            maskKernelType = gpu::KERNEL_TYPE_NEAREST_NEIGHBOR;
        } else {
            throw LSST_EXCEPT(pexExcept::InvalidParameterError, "unknown type of mask warping kernel");
        }
    }
#endif

    if (gpuDetail::TryToSelectCudaDevice(!forceProcessing) == false) {
        return std::pair<int, WarpImageGpuStatus::ReturnCode>(-1, WarpImageGpuStatus::NO_GPU);
    }
        
    int const mainKernelSize = 2 * lanczosKernel.getOrder();
    //do not process if the kernel is too large for allocated GPU local memory
    if (mainKernelSize * 2 > gpu::SIZE_MAX_WARPING_KERNEL) {
        return std::pair<int, WarpImageGpuStatus::ReturnCode>(-1, WarpImageGpuStatus::KERNEL_TOO_LARGE);
    }

    //do not process if the interpolation data is too large to make any speed gains
    if (!forceProcessing && interpLength < 3) {
        return std::pair<int, WarpImageGpuStatus::ReturnCode>(-1, WarpImageGpuStatus::INTERP_LEN_TOO_SMALL);
    }

    int const destWidth = destImage.getWidth();
    int const destHeight = destImage.getHeight();
    pexLog::TTrace<3>("lsst.afw.math.warp", "(GPU) remap image width=%d; height=%d", destWidth, destHeight);

    int const maxCol = destWidth - 1;
    int const maxRow = destHeight - 1;

#ifdef GPU_BUILD
    // Compute borders; use to prevent applying kernel outside of srcImage
    afwGeom::Box2I srcGoodBBox = lanczosKernel.shrinkBBox(srcImage.getBBox(afwImage::LOCAL));
#endif

    int const interpBlkNX = InterpBlkN(destWidth , interpLength);
    int const interpBlkNY = InterpBlkN(destHeight, interpLength);
    //GPU kernel input, will contain: for each interpolation block, all interpolation parameters
    gpuDetail::GpuBuffer2D<gpu::BilinearInterp> srcPosInterp(interpBlkNX, interpBlkNY);

    // calculate values of coordinate transform function
    for (int rowBand = 0; rowBand < interpBlkNY; rowBand++) {
        int row = min(maxRow, (rowBand * interpLength - 1));
        for (int colBand = 0; colBand < interpBlkNX; colBand++) {
            int col = min(maxCol, (colBand * interpLength - 1));
            afwGeom::Point2D srcPos = computeSrcPos(col, row);
            SPoint2 sSrcPos(srcPos);
            sSrcPos = MovePoint(sSrcPos, SVec2(-srcImage.getX0(), -srcImage.getY0()));
            srcPosInterp.Pixel(colBand, rowBand).o =  sSrcPos;
        }
    }

    CalculateInterpolationData(/*in,out*/srcPosInterp, interpLength, destWidth, destHeight);

    int numGoodPixels = 0;

    pexLog::TTrace<3>("lsst.afw.math.warp", "using GPU acceleration, remapping masked image");

#ifdef GPU_BUILD
    int maskKernelSize;
    if (maskKernelType == gpu::KERNEL_TYPE_LANCZOS) {
        maskKernelSize = 2 * dynamic_cast<afwMath::LanczosWarpingKernel const*>(&maskWarpingKernel)->getOrder();
    } else {
        maskKernelSize = 2;
    }
    numGoodPixels = WarpImageGpuWrapper(destImage,
                                        srcImage,
                                        mainKernelSize,
                                        maskKernelType,
                                        maskKernelSize,
                                        srcGoodBBox,
                                        srcPosInterp, interpLength, padValue
                                       );
#endif
    return std::pair<int, WarpImageGpuStatus::ReturnCode>(numGoodPixels, WarpImageGpuStatus::OK);
}

//
// Explicit instantiations
//
/// \cond
#define MASKEDIMAGE(PIXTYPE) afwImage::MaskedImage<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
#define IMAGE(PIXTYPE) afwImage::Image<PIXTYPE>
#define NL /* */

#define INSTANTIATE(DESTIMAGEPIXELT, SRCIMAGEPIXELT) \
    template std::pair<int,WarpImageGpuStatus::ReturnCode> warpImageGPU( \
        IMAGE(DESTIMAGEPIXELT) &destImage, \
        IMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwMath::LanczosWarpingKernel const &warpingKernel, \
        afwMath::SeparableKernel const &maskWarpingKernel, \
        PositionFunctor const &computeSrcPos, \
        int const interpLength, \
        IMAGE(DESTIMAGEPIXELT)::SinglePixel padValue, \
        const bool forceProcessing); NL    \
    template std::pair<int,WarpImageGpuStatus::ReturnCode> warpImageGPU( \
        MASKEDIMAGE(DESTIMAGEPIXELT) &destImage, \
        MASKEDIMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwMath::LanczosWarpingKernel const &warpingKernel, \
        afwMath::SeparableKernel const &maskWarpingKernel, \
        PositionFunctor const &computeSrcPos, \
        int const interpLength, \
        MASKEDIMAGE(DESTIMAGEPIXELT)::SinglePixel padValue, \
        const bool forceProcessing);

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
} //namespace lsst::afw::math::detail ends
