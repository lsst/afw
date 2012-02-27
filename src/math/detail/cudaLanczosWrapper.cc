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

#ifdef IS_GPU_BUILD
    #include <cuda.h>
    #include <cuda_runtime.h>
#endif

#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/geom/Box.h"

#include "lsst/afw/math.h"
#include "lsst/pex/logging/Trace.h"

#include "lsst/afw/gpu/GpuExceptions.h"
#include "lsst/afw/gpu/IsGpuBuild.h"
#include "lsst/afw/gpu/detail/ImageBuffer.h"
#include "lsst/afw/gpu/detail/CudaMemory.h"
#include "lsst/afw/gpu/detail/CudaSelectGpu.h"

#include "lsst/afw/math/detail/CudaLanczos.h"
#include "lsst/afw/math/detail/CudaLanczosWrapper.h"

using namespace std;
using namespace lsst::afw::math::detail::gpu;
using namespace boost::tuples;

namespace mathDetail = lsst::afw::math::detail;
namespace gpuDetail = lsst::afw::gpu::detail;
namespace afwMath = lsst::afw::math;
namespace afwGpu = lsst::afw::gpu;
namespace afwImage = lsst::afw::image;
namespace pexExcept = lsst::pex::exceptions;
namespace pexLog = lsst::pex::logging;
namespace afwGeom = lsst::afw::geom;

#ifdef XXMY_BUILD
extern double timeOrig;
extern double timeTotal;
    extern double timeRows;
        extern double timeColLoop;
        extern double timeSrcPos;
        extern double timeRelArea;
    extern double timeGpuLanczos;
        extern double timeWriteData;
        extern double timeTransferData;
        extern double  timeTransferResult;
        extern double timeKernel;
        extern double timeTransferXX;
        extern double timeWriteback;


void TimeStart(double& clk);
void TimeEnd(double& clk);
void TimeStartSlw(double& clk);
void TimeEndSlw(double& clk);
#else
#define TimeStart(ignore)
#define TimeEnd(ignore)
#define TimeStartSlw(ignore)
#define TimeEndSlw(ignore)
#endif


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
    int InterpBlkN(int size ,int interpLength)
    {
        return CeilDivide(size ,interpLength)+1;
    }

    // calculate the interpolated value given the data for linear interpolation
    gpu::SPoint2 GetInterpolatedValue(afwGpu::detail::ImageBuffer<gpu::BilinearInterp> const & interpBuf,
                                int blkX, int blkY, int subX, int subY
                                )
    {
        gpu::BilinearInterp interp = interpBuf.Pixel(blkX, blkY);
        return interp.Interpolate(subX,subY);
    }

    // calculate the interpolated value given the data for linear interpolation
    gpu::SPoint2 GetInterpolatedValue(afwGpu::detail::ImageBuffer<gpu::BilinearInterp> const & interpBuf,
                                int interpLen, int x, int y
                                )
    {
        int blkX = x/interpLen;
        int blkY = y/interpLen;

        int subX = x%interpLen;
        int subY = y%interpLen;

        return GetInterpolatedValue(interpBuf, blkX, blkY, subX, subY);
    }

    // calculate the number of points falling within the srcGoodBox,
    // given a bilinearily interpolated function on integer range [0,width> x [0, height>
    int NumGoodPixels(afwGpu::detail::ImageBuffer<gpu::BilinearInterp> const & interpBuf,
                      int interpLen, int width, int height, SBox2I srcGoodBox)
    {
        int cnt=0;
        /*for (int row=0; row<height; row++) {
            for (int col=0; col<width; col++) {
                gpu::SPoint2 srcPos=GetInterpolatedValue(interpBuf, interpLen, col+1, row+1);
                if (srcGoodBox.isInsideBox(srcPos)) {
                    cnt++;
                }
            }
        }*/

        int subY=1, blkY=0;
        for (int row=0; row<height; row++, subY++) {
            if (subY >=interpLen) {
                subY-=interpLen;
                blkY++;
                }

            int subX=1, blkX=0;
            gpu::BilinearInterp interp = interpBuf.Pixel(blkX, blkY);
            gpu::LinearInterp lineY=interp.GetLinearInterp(subY);

            for (int col=0; col<width; col++, subX++) {
                if (subX >=interpLen) {
                    subX-=interpLen;
                    blkX++;
                    interp = interpBuf.Pixel(blkX, blkY);
                    lineY=interp.GetLinearInterp(subY);
                    }
                //gpu::SPoint2 srcPos=GetInterpolatedValue(interpBuf, blkX, blkY, subX, subY);
                gpu::SPoint2 srcPos = lineY.Interpolate(subX);
                if (srcGoodBox.isInsideBox(srcPos)) {
                    cnt++;
                }
                }
        }
        return cnt;
    }

#ifdef IS_GPU_BUILD
    // for (plain) Image::
    // allocate CPU and GPU buffers, transfer data and call GPU kernel proxy
    // precondition: order*2 < gpu::cWarpingKernelMaxSize
    template< typename DestPixelT, typename SrcPixelT>
    int WarpImageGpuWrapper(
            afwImage::Image<DestPixelT>     &destImage,
            afwImage::Image<SrcPixelT>const &srcImage,
            int order,
            lsst::afw::geom::Box2I srcBox,
            int kernelCenterX,
            int kernelCenterY,
            lsst::afw::gpu::detail::ImageBuffer<SBox2I> const& srcBlk,
            lsst::afw::gpu::detail::ImageBuffer<BilinearInterp> const& srcPosInterp,
            int interpLength
            )
    {
        //TimeStart(timeWriteData);
        typedef typename afwImage::Image<DestPixelT> DestImageT;

        typename DestImageT::SinglePixel const edgePixel = afwMath::edgePixel<DestImageT>(
            typename afwImage::detail::image_traits<DestImageT>::image_category()
        );

        PixelIVM<DestPixelT> edgePixelGpu;
        edgePixelGpu.img=edgePixel;
        edgePixelGpu.var=-1;
        edgePixelGpu.msk=-1;

        const int destWidth = destImage.getWidth();
        const int destHeight = destImage.getHeight();
        gpuDetail::GpuMemOwner<DestPixelT> destBufImgGpu;
        gpuDetail::GpuMemOwner<SrcPixelT> srcBufImgGpu;
        gpuDetail::GpuMemOwner<SBox2I> srcBlkGpu;
        gpuDetail::GpuMemOwner<BilinearInterp> srcPosInterpGpu;
        //TimeEnd(timeWriteData);

        TimeStart(timeTransferData);
        ImageDataPtr<DestPixelT> destImgGpu;
        destImgGpu.strideImg=destBufImgGpu.AllocImageBaseBuffer(destImage);
        //destImgGpu.strideImg=destBufImgGpu.TransferFromImageBase(*dstImage.getImage()    ,destBufImgGpu);
        if (destBufImgGpu.ptr == NULL)  {
            throw LSST_EXCEPT(afwGpu::GpuMemoryException, "Not enough memory on GPU for output image");
        }
        destImgGpu.img=destBufImgGpu.ptr;
        destImgGpu.var=NULL;
        destImgGpu.msk=NULL;
        destImgGpu.width=destWidth;
        destImgGpu.height=destHeight;

        ImageDataPtr<SrcPixelT> srcImgGpu;
        srcImgGpu.strideImg=srcBufImgGpu.TransferFromImageBase(srcImage);
        if (srcBufImgGpu.ptr == NULL)  {
            throw LSST_EXCEPT(afwGpu::GpuMemoryException, "Not enough memory on GPU for input image");
        }
        srcImgGpu.img=srcBufImgGpu.ptr;
        srcImgGpu.var=NULL;
        srcImgGpu.msk=NULL;
        srcImgGpu.width=srcImage.getWidth();
        srcImgGpu.height=srcImage.getHeight();

        srcBlkGpu.Transfer(srcBlk);
        if (srcBlkGpu.ptr == NULL)  {
            throw LSST_EXCEPT(afwGpu::GpuMemoryException, "Not enough memory on GPU for source block sizes");
        }
        srcPosInterpGpu.Transfer(srcPosInterp);
        if (srcBlkGpu.ptr == NULL)  {
            throw LSST_EXCEPT(afwGpu::GpuMemoryException,
                              "Not enough memory on GPU for interpolation data for coorinate transformation");
        }

        SBox2I srcBoxConv(srcBox.getMinX(), srcBox.getMinY(), srcBox.getMaxX()+1,srcBox.getMaxY()+1);
        TimeEnd(timeTransferData);

        TimeStart(timeKernel);

        WarpImageGpuCallKernel(false,
                               destImgGpu, srcImgGpu,
                               order,
                               srcBoxConv,
                               kernelCenterX,
                               kernelCenterY,
                               edgePixelGpu,
                               srcBlkGpu.ptr,
                               srcPosInterpGpu.ptr, interpLength
                               );

        TimeStart(timeWriteData);
        int numGoodPixels=NumGoodPixels(srcPosInterp, interpLength, destWidth, destHeight, srcBoxConv);
        TimeEnd(timeWriteData);

        TimeStart(timeWriteback);
        cudaThreadSynchronize();
        TimeEnd(timeWriteback);
        if (cudaGetLastError() != cudaSuccess) {
            throw LSST_EXCEPT(afwGpu::GpuRuntimeErrorException, "GPU calculation failed to run");
            }

        TimeEnd(timeKernel);

        TimeStart(timeTransferResult);
        destBufImgGpu.CopyToImageBase(destImage);
        TimeEnd(timeTransferResult);

        //TimeStart(timeWriteback);
        //destBufImg.CopyToImage(destImage,0,0);
        //TimeEnd(timeWriteback);

        return numGoodPixels;
    }

    // for MaskedImage::
    // allocate CPU and GPU buffers, transfer data and call GPU kernel proxy
    // precondition: order*2 < gpu::cWarpingKernelMaxSize
    template< typename DestPixelT, typename SrcPixelT>
    int WarpImageGpuWrapper(
            afwImage::MaskedImage<DestPixelT>      &dstImage,
            afwImage::MaskedImage<SrcPixelT>const  &srcImage,
            int order,
            lsst::afw::geom::Box2I srcBox,
            int kernelCenterX,
            int kernelCenterY,
            lsst::afw::gpu::detail::ImageBuffer<SBox2I> const& srcBlk,
            lsst::afw::gpu::detail::ImageBuffer<BilinearInterp> const& srcPosInterp,
            int interpLength
            )
    {
        TimeStart(timeWriteData);

        typedef typename afwImage::MaskedImage<DestPixelT> DestImageT;

        typename DestImageT::SinglePixel const edgePixel = afwMath::edgePixel<DestImageT>(
            typename afwImage::detail::image_traits<DestImageT>::image_category()
        );

        PixelIVM<DestPixelT> edgePixelGpu;
        edgePixelGpu.img=edgePixel.image();
        edgePixelGpu.var=edgePixel.variance();
        edgePixelGpu.msk=edgePixel.mask();

        const int destWidth = dstImage.getWidth();
        const int destHeight = dstImage.getHeight();

        gpuDetail::GpuMemOwner<DestPixelT> destBufImgGpu;
        gpuDetail::GpuMemOwner<VarPixel>   destBufVarGpu;
        gpuDetail::GpuMemOwner<MskPixel>   destBufMskGpu;

        gpuDetail::GpuMemOwner<SrcPixelT> srcBufImgGpu;
        gpuDetail::GpuMemOwner<VarPixel>  srcBufVarGpu;
        gpuDetail::GpuMemOwner<MskPixel>  srcBufMskGpu;

        gpuDetail::GpuMemOwner<SBox2I> srcBlkGpu;
        gpuDetail::GpuMemOwner<BilinearInterp> srcPosInterpGpu;

        //typename afwImage::MaskedImage<DestPixelT>::Pixel p(222, 333, 444);
        //dstImage=p;

        TimeEnd(timeWriteData);

        TimeStart(timeTransferData);

        ImageDataPtr<DestPixelT> destImgGpu;
        destImgGpu.strideImg=destBufImgGpu.AllocImageBaseBuffer(*dstImage.getImage());
        destImgGpu.strideVar=destBufVarGpu.AllocImageBaseBuffer(*dstImage.getVariance());
        destImgGpu.strideMsk=destBufMskGpu.AllocImageBaseBuffer(*dstImage.getMask());
        /*
        destImgGpu.strideImg=destBufImgGpu.TransferFromImageBase(*dstImage.getImage());
        destImgGpu.strideVar=destBufVarGpu.TransferFromImageBase(*dstImage.getVariance());
        destImgGpu.strideMsk=destBufMskGpu.TransferFromImageBase(*dstImage.getMask());*/
        if (destBufImgGpu.ptr == NULL)  {
            throw LSST_EXCEPT(afwGpu::GpuMemoryException, "Not enough memory on GPU for input image");
        }
        if (destBufVarGpu.ptr == NULL)  {
            throw LSST_EXCEPT(afwGpu::GpuMemoryException, "Not enough memory on GPU for input variance");
        }
        if (destBufMskGpu.ptr == NULL)  {
            throw LSST_EXCEPT(afwGpu::GpuMemoryException, "Not enough memory on GPU for input mask");
        }
        destImgGpu.img=destBufImgGpu.ptr;
        destImgGpu.var=destBufVarGpu.ptr;
        destImgGpu.msk=destBufMskGpu.ptr;
        destImgGpu.width=destWidth;
        destImgGpu.height=destHeight;

        ImageDataPtr<SrcPixelT> srcImgGpu;
        srcImgGpu.strideImg=srcBufImgGpu.TransferFromImageBase(*srcImage.getImage());
        if (srcBufImgGpu.ptr == NULL)  {
            throw LSST_EXCEPT(afwGpu::GpuMemoryException, "Not enough memory on GPU for input image");
        }
        srcImgGpu.strideVar=srcBufVarGpu.TransferFromImageBase(*srcImage.getVariance());
        if (srcBufVarGpu.ptr == NULL)  {
            throw LSST_EXCEPT(afwGpu::GpuMemoryException, "Not enough memory on GPU for input variance");
        }
        srcImgGpu.strideMsk=srcBufMskGpu.TransferFromImageBase(*srcImage.getMask());
        if (srcBufMskGpu.ptr == NULL)  {
            throw LSST_EXCEPT(afwGpu::GpuMemoryException, "Not enough memory on GPU for input mask");
        }

        srcImgGpu.img=srcBufImgGpu.ptr;
        srcImgGpu.var=srcBufVarGpu.ptr;
        srcImgGpu.msk=srcBufMskGpu.ptr;
        srcImgGpu.width=srcImage.getWidth();
        srcImgGpu.height=srcImage.getHeight();

        srcBlkGpu.Transfer(srcBlk);
        if (srcBlkGpu.ptr == NULL)  {
            throw LSST_EXCEPT(afwGpu::GpuMemoryException, "Not enough memory on GPU for source block sizes");
        }
        srcPosInterpGpu.Transfer(srcPosInterp);
        if (srcBlkGpu.ptr == NULL)  {
            throw LSST_EXCEPT(afwGpu::GpuMemoryException,
                              "Not enough memory on GPU for interpolation data for coorinate transformation");
        }

        SBox2I srcBoxConv(srcBox.getMinX(), srcBox.getMinY(), srcBox.getMaxX()+1,srcBox.getMaxY()+1);
        TimeEnd(timeTransferData);

        //printf("MASKED KERNEL!!\n");
        TimeStart(timeKernel);

        WarpImageGpuCallKernel(true,
                               destImgGpu, srcImgGpu,
                               order,
                               srcBoxConv,
                               kernelCenterX,
                               kernelCenterY,
                               edgePixelGpu,
                               srcBlkGpu.ptr,
                               srcPosInterpGpu.ptr, interpLength
                               );
        int numGoodPixels=NumGoodPixels(srcPosInterp, interpLength, destWidth, destHeight, srcBoxConv);

        cudaThreadSynchronize();
        if (cudaGetLastError() != cudaSuccess) {
            throw LSST_EXCEPT(afwGpu::GpuRuntimeErrorException, "GPU calculation failed to run");
            }
        TimeEnd(timeKernel);

        TimeStart(timeTransferResult);
        destBufImgGpu.CopyToImageBase(*dstImage.getImage());
        destBufVarGpu.CopyToImageBase(*dstImage.getVariance());
        destBufMskGpu.CopyToImageBase(*dstImage.getMask());
        TimeEnd(timeTransferResult);

        //TimeStart(timeWriteback);
        //lsst::afw::math::detail::CopyToMaskedImage(dstImage, 0, 0, destBufImg, destBufVar, destBufMsk);
        //TimeEnd(timeWriteback);

        return numGoodPixels;
    }
#endif //IS_GPU_BUILD

    // Calculate bilinear interpolation data based on given function values
    // input:
    //    srcPosInterp - contains values of original function at a mesh of equally distanced points
    //                  the values are stored in .o member
    //    interpLength - distance between points
    //    destWidth, destHeight - size of function domain
    // output:
    //    srcPosInterp - all members are calculated and set, ready to calculate interpolation values
    void CalculateInterpolationData(gpuDetail::ImageBuffer<BilinearInterp>& srcPosInterp, int interpLength,
                                    int destWidth, int destHeight)
    {
        const int interpBlkNX = InterpBlkN(destWidth ,interpLength);
        const int interpBlkNY = InterpBlkN(destHeight,interpLength);

        for (int row=-1, rowBand = 0; rowBand<interpBlkNY-1; row+=interpLength, rowBand++) {
            const double invInterpLen= 1.0/interpLength;
            const double invInterpLenRow= row+interpLength <= destHeight-1?
                                            invInterpLen : 1.0/(destHeight-1-row);

            for (int col=-1, colBand = 0; colBand<interpBlkNX-1; col+=interpLength, colBand++) {

                const SPoint2 p11 = srcPosInterp.Pixel(colBand  ,rowBand  ).o;
                const SPoint2 p12 = srcPosInterp.Pixel(colBand+1,rowBand  ).o;
                const SPoint2 p21 = srcPosInterp.Pixel(colBand  ,rowBand+1).o;
                const SPoint2 p22 = srcPosInterp.Pixel(colBand+1,rowBand+1).o;
                const SVec2 band_dY  = SVec2(p11, p21);
                const SVec2 band_d0X = SVec2(p11, p12);
                const SVec2 band_d1X = SVec2(p21, p22);
                const SVec2 band_ddX = VecMul( VecSub(band_d1X,band_d0X), invInterpLenRow);

                const double invInterpLenCol= col+interpLength <= destWidth-1?
                                            invInterpLen : 1.0/(destWidth-1-col);

                BilinearInterp lin = srcPosInterp.Pixel(colBand, rowBand); //sets lin.o
                lin.deltaY = VecMul(band_dY , invInterpLenRow);
                lin.d0X    = VecMul(band_d0X, invInterpLenCol);
                lin.ddX    = VecMul(band_ddX, invInterpLenCol);
                srcPosInterp.Pixel(colBand, rowBand) = lin;

                // partially fill the last column and row, too
                if (colBand==interpBlkNX-2){
                    srcPosInterp.Pixel(interpBlkNX-1, rowBand).deltaY =
                        VecMul( SVec2(p12, p22), invInterpLenRow);
                }
                if (rowBand==interpBlkNY-2){
                    srcPosInterp.Pixel(colBand, interpBlkNY-1).d0X =
                        VecMul( SVec2(p21, p22), invInterpLenCol);
                    }
            }
        }
    }

} //local namespace ends

//part of public interface
template<typename DestImageT, typename SrcImageT>
std::pair<int,bool> warpImageGPU(
    DestImageT &destImage,              ///< remapped %image
    afwImage::Wcs const &destWcs,       ///< WCS of remapped %image
    SrcImageT const &srcImage,          ///< source %image
    afwImage::Wcs const &srcWcs,        ///< WCS of source %image
    afwMath::SeparableKernel &warpingKernel,     ///< warping kernel; determines warping algorithm
    int const interpLength,              ///< Distance over which WCS can be linearily interpolated
                                        ///< must be >0
    lsst::afw::gpu::DevicePreference devPref
    )
{
    afwMath::LanczosWarpingKernel const* lanKernel=
            dynamic_cast<afwMath::LanczosWarpingKernel const*>(&warpingKernel);
    if (lanKernel==NULL)
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                    "Kernel for GPU acceleration must be Lanczos kernel");

    if (interpLength < 1) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                    "GPU accelerated warping must use interpolation");
    }

#ifndef IS_GPU_BUILD
    throw LSST_EXCEPT(afwGpu::GpuRuntimeErrorException, "Afw not compiled with GPU support");
#else
    if (gpuDetail::TryToSelectCudaDevice(devPref) == false)
        return std::pair<int,bool>(-1,false);

    //do not process if the kernel is too large for allocated GPU local memory
    const int order=lanKernel->getOrder();
    if (order*2>gpu::cWarpingKernelMaxSize)
        return std::pair<int,bool>(-1,false);

    //do not process if the interpolation data is too large to make any speed gains
    if (devPref!=lsst::afw::gpu::USE_GPU && interpLength < 3) {
        return std::pair<int,bool>(-1,false);
    }

    // Compute borders; use to prevent applying kernel outside of srcImage
    int const kernelCtrX = warpingKernel.getCtrX();
    int const kernelCtrY = warpingKernel.getCtrY();

    int const destWidth = destImage.getWidth();
    int const destHeight = destImage.getHeight();
    int const maxCol = destWidth - 1;
    int const maxRow = destHeight - 1;
    afwGeom::Point2D const destXY0(destImage.getXY0());

    typedef typename DestImageT::SinglePixel DestPixelT;
    typedef typename  SrcImageT::SinglePixel SrcPixelT;

    afwGeom::Box2I srcGoodBBox = warpingKernel.shrinkBBox(srcImage.getBBox(afwImage::LOCAL));

    const int interpBlkNX = InterpBlkN(destWidth ,interpLength);
    const int interpBlkNY = InterpBlkN(destHeight,interpLength);
    //GPU kernel input, will contain: for each interpolation block, all interpolation parameters
    gpuDetail::ImageBuffer<BilinearInterp> srcPosInterp(interpBlkNX, interpBlkNY);

    // calculate values of coordinate transform function
    TimeStart(timeRows);
    TimeStart(timeSrcPos);
    for (int rowBand = 0; rowBand<interpBlkNY;rowBand++) {
        int row=min(maxRow,(rowBand*interpLength-1));
        for (int colBand = 0; colBand<interpBlkNX; colBand++) {
            int col=min(maxCol,(colBand*interpLength-1));
            afwGeom::Point2D srcPos = computeSrcPos(col, row, destXY0, destWcs, srcWcs);
            SPoint2 sSrcPos(srcPos);
            sSrcPos=MovePoint(sSrcPos, SVec2(-srcImage.getX0(),-srcImage.getY0()));
            srcPosInterp.Pixel(colBand, rowBand).o =  sSrcPos;
        }
    }
    TimeEnd(timeSrcPos);

    TimeStart(timeRelArea);

    CalculateInterpolationData(/*in,out*/srcPosInterp, interpLength, destWidth, destHeight);

    TimeEnd(timeRelArea);
    // calculates dimensions of partitions of destination image to GPU blocks
    // each block is handled by one GPU multiprocessor
    const int gpuBlockSizeX=gpu::cWarpingBlockSizeX;
    const int gpuBlockSizeY=gpu::cWarpingBlockSizeY;
    const int gpuBlockXN = CeilDivide(destWidth, gpuBlockSizeX);
    const int gpuBlockYN = CeilDivide(destHeight, gpuBlockSizeY);
    //***UNUSED*** GPU input, will contain: for each gpu block, the box specifying the required source image data
    gpuDetail::ImageBuffer<gpu::SBox2I> srcBlk(gpuBlockXN, gpuBlockYN);

    TimeEnd(timeRows);

    TimeStart(timeGpuLanczos);
    int numGoodPixels = 0;

    numGoodPixels = WarpImageGpuWrapper(destImage,
                                    srcImage,
                                    order,
                                    srcGoodBBox,
                                    kernelCtrX, kernelCtrY,
                                    srcBlk, srcPosInterp, interpLength
                                    );

    TimeEnd(timeGpuLanczos);

    return std::pair<int,bool>(numGoodPixels,true);
#endif //IS_GPU_BUILD
}

//
// Explicit instantiations
//
/// \cond
#define MASKEDIMAGE(PIXTYPE) afwImage::MaskedImage<PIXTYPE, afwImage::MaskPixel, afwImage::VariancePixel>
#define IMAGE(PIXTYPE) afwImage::Image<PIXTYPE>
#define NL /* */

#define INSTANTIATE(DESTIMAGEPIXELT, SRCIMAGEPIXELT) \
    template std::pair<int,bool> warpImageGPU( \
        IMAGE(DESTIMAGEPIXELT) &destImage, \
        afwImage::Wcs const &destWcs, \
        IMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwImage::Wcs const &srcWcs, \
        afwMath::SeparableKernel &warpingKernel, \
        int const interpLength, \
        lsst::afw::gpu::DevicePreference devPref); NL    \
    template std::pair<int,bool> warpImageGPU( \
        MASKEDIMAGE(DESTIMAGEPIXELT) &destImage, \
        afwImage::Wcs const &destWcs, \
        MASKEDIMAGE(SRCIMAGEPIXELT) const &srcImage, \
        afwImage::Wcs const &srcWcs, \
        afwMath::SeparableKernel &warpingKernel, \
        int const interpLength, \
        lsst::afw::gpu::DevicePreference devPref);

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

}}}} //namespace lsst::afw::math::detail ends
