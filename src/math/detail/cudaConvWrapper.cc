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
 * Functions in this file are used to allocate necessary buffers,
 * transfer data from and to GPU memory, and to set up and perform convolution.
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

#ifndef GPU_BUILD

namespace lsst {
namespace afw {
namespace math {
namespace detail {


void TestGpuKernel(int& ret1, int& ret2) {
    ret1 = 0;
    ret2 = 0;
}

bool IsSufficientSharedMemoryAvailable_ForImgBlock(int filterW, int filterH, int pixSize)
{
    return false;
}
bool IsSufficientSharedMemoryAvailable_ForImgAndMaskBlock(int filterW, int filterH, int pixSize)
{
    return false;
}
bool IsSufficientSharedMemoryAvailable_ForSfn(int order, int kernelN)
{
    return false;
}

}
}
}
}

#else

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "lsst/afw/image/LsstImageTypes.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/FunctionLibrary.h"

#include "lsst/afw/gpu/detail/GpuBuffer2D.h"
#include "lsst/afw/math/detail/convCUDA.h"
#include "lsst/afw/math/detail/cudaConvWrapper.h"
#include "lsst/afw/gpu/detail/CudaQueryDevice.h"
#include "lsst/afw/math/detail/Convolve.h"
#include "lsst/afw/gpu/detail/CudaSelectGpu.h"
#include "lsst/afw/gpu/detail/CudaMemory.h"

using namespace std;
using namespace lsst::afw::gpu;
using namespace lsst::afw::gpu::detail;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;
namespace mathDetailGpu = lsst::afw::math::detail::gpu;



namespace lsst {
namespace afw {
namespace math {
namespace detail {

namespace {
const int shMemBytesUsed = 200;
}


// Returns true if there is sufficient shared memory for loading an image block,
// where image block includes including filter frame.
bool IsSufficientSharedMemoryAvailable_ForImgBlock(int filterW, int filterH, int pixSize)
{
    int shMemSize = GetCudaCurSMSharedMemorySize();
    int bufferSize = (filterW + blockSizeX - 1) * (filterH + blockSizeY - 1) * pixSize;

    return shMemSize - shMemBytesUsed - bufferSize > 0;
}

// Returns true if there is sufficient shared memory for loading an image block,
// and acommpanying block of mask data (mask image block),
// where image block and mask image block include including filter frame.
bool IsSufficientSharedMemoryAvailable_ForImgAndMaskBlock(int filterW, int filterH, int pixSize)
{
    int shMemSize = GetCudaCurSMSharedMemorySize();
    int imgBufferSize = (filterW + blockSizeX - 1) * (filterH + blockSizeY - 1) * pixSize;
    int mskBufferSize = (filterW + blockSizeX - 1) * (filterH + blockSizeY - 1) * sizeof(MskPixel);

    int memRemaining = shMemSize - shMemBytesUsed - imgBufferSize - mskBufferSize ;

    return memRemaining > 0;
}

// Returns true if there is sufficient shared memory for loading
// parameters and temporary values for Chebyshev function and normalization
bool IsSufficientSharedMemoryAvailable_ForSfn(int order, int kernelN)
{
    int shMemSize = GetCudaCurSMSharedMemorySize();

    const int coeffN = order + 1;
    const int coeffPadding = coeffN + 1 - (coeffN % 2);
    int paramN = (order + 1) * (order + 2) / 2;

    int yCoeffsAll = coeffPadding * blockSizeX * sizeof(double);
    int xChebyAll = coeffPadding * blockSizeX * sizeof(double);
    int smemParams = paramN * sizeof(double);

    int  smemKernelSum = kernelN * sizeof(double);
    int  smemSfnPtr   = kernelN * sizeof(double*);

    int memRemainingSfn = shMemSize - shMemBytesUsed - yCoeffsAll - xChebyAll - smemParams ;
    int memRemainingNorm = shMemSize - shMemBytesUsed - smemKernelSum - smemSfnPtr;

    return min(memRemainingSfn, memRemainingNorm) > 0;
}

// This function decides on the best GPU block count
// uses simple heuristics (not to much blocks and not too many)
// but guarantees that number of blocks will be a multiple of number of multiprocessors
int CalcBlockCount(int multiprocCount)
{
    if (multiprocCount < 12)  return multiprocCount * 4;
    if (multiprocCount < 24)  return multiprocCount * 2;
    return multiprocCount;
}


// calls test gpu kernel
// should return 5 and 8 in ret1 and ret2
void TestGpuKernel(int& ret1, int& ret2)
{
    int res[2];

    GpuMemOwner<int> resGpu;
    resGpu.Alloc(2);

    gpu::CallTestGpuKernel(resGpu.ptr);

    resGpu.CopyFromGpu(res);

    ret1 = res[0];
    ret2 = res[1];
}

namespace {

//calculates sum of each image in 'images' vector
template <typename ResultT, typename InT>
vector<ResultT> SumsOfImages(const vector< GpuBuffer2D<InT> >&  images)
{
    int n = int(images.size());
    vector<ResultT> sum(n);
    for (int i = 0; i < n; i++) {
        ResultT totalSum = 0;
        int h = images[i].height;
        int w = images[i].width;

        for (int y = 0; y < h; y++) {
            ResultT rowSum = 0;
            for (int x = 0; x < w; x++) {
                rowSum += images[i].Pixel(x, y);
            }
            totalSum += rowSum;
        }
        sum[i] = totalSum;
    }
    return sum;
}

/**
    Convolves given inImage with linear combination kernel.

    Calculates:
        - the sFn (spatial function) values, output in sFnValGPUPtr and sFnValGPU
        - if doNormalize is true, will also compute normalization coefficients.
            Normalization coefficients will be placed in normGPU buffer.
        - the result of convolution with LinearCombination kernel (given
            by sFn) in out image
        - sFnValGPUPtr and normGPU reside in GPU memory

    Basis kernels, given by basisKernelsListGPU, must have been transfered to GPU memory previously.
    Other GPU output data parameters (sFnValGPUPtr, sFnValGPU, normGPU) must be already allocated.
*/
template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionImage_LC_Img(
    const GpuBuffer2D<InPixelT>& inImage,
    const vector<double>& colPos,
    const vector<double>& rowPos,
    const std::vector< afwMath::Kernel::SpatialFunctionPtr >& sFn,
    const vector<double*>& sFnValGPUPtr, //output
    double** sFnValGPU,    //output
    SpatialFunctionType_t sfType,
    GpuBuffer2D<OutPixelT>&  outImage, //output
    KerPixel*   basisKernelsListGPU,
    int kernelW, int kernelH,
    const vector<double>&   basisKernelSums,   //input
    double* normGPU,   //output
    bool doNormalize
)
{
    const int kernelN = sFn.size();

    //transfer input image
    GpuMemOwner<InPixelT > inImageGPU;
    inImageGPU.Transfer(inImage);
    if (inImageGPU.ptr == NULL)  {
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU for input image");
    }
    // allocate output image planes on GPU
    GpuMemOwner<OutPixelT> outImageGPU;
    outImageGPU.Alloc( outImage.Size());
    if (outImageGPU.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU for output image");
    }
    //transfer coordinate tranform data
    GpuMemOwner<double> colPosGPU_Owner;
    colPosGPU_Owner.TransferVec(colPos);
    if (colPosGPU_Owner.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryError,
                          "Not enough memory on GPU for row coordinate tranformation data");
    }
    GpuMemOwner<double> rowPosGPU_Owner;
    rowPosGPU_Owner.TransferVec(rowPos);
    if (rowPosGPU_Owner.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryError,
                          "Not enough memory on GPU for column coordinate tranformation data");
    }
    vector< double* >                     sFnParamsGPUPtr(kernelN);
    vector< GpuMemOwner<double> >    sFnParamsGPU_Owner(kernelN);

    //transfer sfn parameters to GPU
    for (int i = 0; i < kernelN; i++) {
        std::vector<double> spatialParams = sFn[i]->getParameters();
        sFnParamsGPUPtr[i] = sFnParamsGPU_Owner[i].TransferVec(spatialParams);
        if (sFnParamsGPUPtr[i] == NULL) {
            throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU for spatial function parameters");
        }
    }

    int shMemSize = GetCudaCurSMSharedMemorySize() - shMemBytesUsed;

    for (int i = 0; i < kernelN; i++)
    {
        cudaGetLastError(); //clear error status

        if (sfType == sftPolynomial) {
            const afwMath::PolynomialFunction2<double>* polySfn =
                dynamic_cast<const afwMath::PolynomialFunction2<double>*>( sFn[i].get() );

            gpu::Call_PolynomialImageValues(
                sFnValGPUPtr[i], outImage.width, outImage.height,
                polySfn->getOrder(),
                sFnParamsGPU_Owner[i].ptr,
                rowPosGPU_Owner.ptr,
                colPosGPU_Owner.ptr,
                shMemSize
            );

            //cudaThreadSynchronize();
            cudaError_t cuda_err = cudaGetLastError();
            if (cuda_err != cudaSuccess) {
                throw LSST_EXCEPT(GpuRuntimeError, "GPU calculation failed to run");
            }
        }
        if (sfType == sftChebyshev) {
            const afwMath::Chebyshev1Function2<double>* chebSfn =
                dynamic_cast<const afwMath::Chebyshev1Function2<double>*>( sFn[i].get() );

            lsst::afw::geom::Box2D const xyRange = chebSfn->getXYRange();

            gpu::Call_ChebyshevImageValues(
                sFnValGPUPtr[i], outImage.width, outImage.height,
                chebSfn->getOrder(),
                sFnParamsGPU_Owner[i].ptr,
                rowPosGPU_Owner.ptr,
                colPosGPU_Owner.ptr,
                xyRange.getMinX(), xyRange.getMinY(), xyRange.getMaxX(), xyRange.getMaxY(),
                shMemSize
            );

            //cudaThreadSynchronize();
            cudaError_t cuda_err = cudaGetLastError();
            if (cuda_err != cudaSuccess) {
                throw LSST_EXCEPT(GpuRuntimeError, "GPU calculation failed to run");
            }
        }
    }
    cudaThreadSynchronize();
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        throw LSST_EXCEPT(GpuRuntimeError, "GPU calculation failed to run");
    }

    int blockN = CalcBlockCount( GetCudaCurSMCount());

    //transfer basis kernel sums
    if (doNormalize) {
        GpuMemOwner<double> basisKernelSumsGPU;
        basisKernelSumsGPU.TransferVec(basisKernelSums);

        bool isDivideByZero = false;
        GpuMemOwner<bool> isDivideByZeroGPU;
        isDivideByZeroGPU.Transfer(&isDivideByZero, 1);

        gpu::Call_NormalizationImageValues(
            normGPU, outImage.width, outImage.height,
            sFnValGPU, kernelN,
            basisKernelSumsGPU.ptr,
            isDivideByZeroGPU.ptr,
            blockN,
            shMemSize
        );
        cudaThreadSynchronize();
        if (cudaGetLastError() != cudaSuccess) {
            throw LSST_EXCEPT(GpuRuntimeError, "GPU calculation failed to run");
        }
        CopyFromGpu<bool>(&isDivideByZero, isDivideByZeroGPU.ptr, 1);
        if (isDivideByZero) {
            throw LSST_EXCEPT(pexExcept::OverflowError, "Cannot normalize; kernel sum is 0");
        }
    }

    cudaGetLastError(); //clear error status
    mathDetailGpu::Call_ConvolutionKernel_LC_Img(
        inImageGPU.ptr, inImage.width, inImage.height,
        basisKernelsListGPU, kernelN,
        kernelW, kernelH,
        &sFnValGPU[0],
        normGPU,
        outImageGPU.ptr,
        blockN,
        shMemSize
    );
    cudaThreadSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        throw LSST_EXCEPT(GpuRuntimeError, "GPU calculation failed to run");
    }

    CopyFromGpu(outImage.img, outImageGPU.ptr, outImage.Size() );

}

} //local namespace ends

template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionImage_LinearCombinationKernel(
    GpuBuffer2D<InPixelT>& inImage,
    vector<double> colPos,
    vector<double> rowPos,
    std::vector< afwMath::Kernel::SpatialFunctionPtr > sFn,
    GpuBuffer2D<OutPixelT>&                outImage,
    std::vector< GpuBuffer2D<KerPixel> >&  basisKernels,
    SpatialFunctionType_t sfType,
    bool doNormalize
)
{
    assert(basisKernels.size() == sFn.size());

    int outWidth = outImage.width;
    int outHeight = outImage.height;

    const int kernelN = sFn.size();
    const int kernelW = basisKernels[0].width;
    const int kernelH = basisKernels[0].height;
    const int kernelSize = kernelW * kernelH;

    for (int i = 0; i < kernelN; i++) {
        assert(kernelW == basisKernels[i].width);
        assert(kernelH == basisKernels[i].height);
    }

    // transfer array of basis kernels on GPU
    GpuMemOwner<KerPixel > basisKernelsGPU;
    basisKernelsGPU.Alloc(kernelSize * kernelN);

    for (int i = 0; i < kernelN; i++) {
        KerPixel* kernelBeg = basisKernelsGPU.ptr + (kernelSize * i);
        CopyToGpu(kernelBeg,
                       basisKernels[i].img,
                       kernelSize
                      );
    }

    // allocate array of spatial function value images on GPU
    vector< double* >                     sFnValGPUPtr(kernelN);
    vector< GpuMemOwner<double > >   sFnValGPU_Owner(kernelN);

    for (int i = 0; i < kernelN; i++) {
        sFnValGPUPtr[i] = sFnValGPU_Owner[i].Alloc(outWidth * outHeight);
        if (sFnValGPUPtr[i] == NULL) {
            throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU for spatial function values");
        }
    }
    GpuMemOwner<double*> sFnValGPU;
    sFnValGPU.TransferVec(sFnValGPUPtr);

    GpuMemOwner<double > normGPU_Owner;
    vector<double> basisKernelSums(kernelN);
    if (doNormalize) {
        //allocate normalization coeficients
        normGPU_Owner.Alloc(outWidth * outHeight);
        if (normGPU_Owner.ptr == NULL) {
            throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU for normalization coefficients");
        }

        //calculate basis kernel sums
        basisKernelSums = SumsOfImages<double, KerPixel>(basisKernels);
    }

    GPU_ConvolutionImage_LC_Img(
        inImage,
        colPos, rowPos,
        sFn,
        sFnValGPUPtr, //output
        sFnValGPU.ptr, //output
        sfType,
        outImage, //output
        basisKernelsGPU.ptr,
        kernelW, kernelH,
        basisKernelSums,   //input
        normGPU_Owner.ptr,   //output
        doNormalize
    );

}

#define INSTANTIATE_GPU_ConvolutionImage_LinearCombinationKernel(OutPixelT,InPixelT)  \
        template void GPU_ConvolutionImage_LinearCombinationKernel<OutPixelT,InPixelT>( \
                    GpuBuffer2D<InPixelT>& inImage, \
                    vector<double> colPos, \
                    vector<double> rowPos, \
                    std::vector< afwMath::Kernel::SpatialFunctionPtr > sFn, \
                    GpuBuffer2D<OutPixelT>&                outImage, \
                    std::vector< GpuBuffer2D<KerPixel> >&  basisKernels, \
                    SpatialFunctionType_t sfType, \
                    bool doNormalize \
                    );

template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionMI_LinearCombinationKernel(
    GpuBuffer2D<InPixelT>& inImageImg,
    GpuBuffer2D<VarPixel>& inImageVar,
    GpuBuffer2D<MskPixel>& inImageMsk,
    vector<double> colPos,
    vector<double> rowPos,
    std::vector< afwMath::Kernel::SpatialFunctionPtr > sFn,
    GpuBuffer2D<OutPixelT>&                outImageImg,
    GpuBuffer2D<VarPixel>&                 outImageVar,
    GpuBuffer2D<MskPixel>&                 outImageMsk,
    std::vector< GpuBuffer2D<KerPixel> >&  basisKernels,
    SpatialFunctionType_t sfType,
    bool doNormalize
)
{
    assert(basisKernels.size() == sFn.size());
    assert(outImageImg.width == outImageVar.width);
    assert(outImageImg.width == outImageMsk.width);
    assert(outImageImg.height == outImageVar.height);
    assert(outImageImg.height == outImageMsk.height);

    int outWidth = outImageImg.width;
    int outHeight = outImageImg.height;

    const int kernelN = sFn.size();
    const int kernelW = basisKernels[0].width;
    const int kernelH = basisKernels[0].height;
    const int kernelSize = kernelW * kernelH;

    for (int i = 0; i < kernelN; i++) {
        assert(kernelW == basisKernels[i].width);
        assert(kernelH == basisKernels[i].height);
    }

    // transfer basis kernels to GPU
    GpuMemOwner<KerPixel > basisKernelsGPU;
    basisKernelsGPU.Alloc(kernelSize * kernelN);

    for (int i = 0; i < kernelN; i++) {
        KerPixel* kernelBeg = basisKernelsGPU.ptr + (kernelSize * i);
        CopyToGpu(kernelBeg,
                       basisKernels[i].img,
                       kernelSize
                      );
    }

    //alloc sFn images on GPU
    vector< double* >                     sFnValGPUPtr(kernelN);
    vector< GpuMemOwner<double > >   sFnValGPU_Owner(kernelN);

    for (int i = 0; i < kernelN; i++) {
        sFnValGPUPtr[i] = sFnValGPU_Owner[i].Alloc(outWidth * outHeight);
        if (sFnValGPUPtr[i] == NULL) {
            throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU for spatial function values");
        }
    }
    GpuMemOwner<double*> sFnValGPU;
    sFnValGPU.TransferVec(sFnValGPUPtr);

    //allocate normalization coeficients image on GPU
    GpuMemOwner<double > normGPU_Owner;
    std::vector<KerPixel> basisKernelSums(kernelN);
    if (doNormalize) {
        //allocate normalization coeficients
        normGPU_Owner.Alloc(outWidth * outHeight);
        if (normGPU_Owner.ptr == NULL) {
            throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU for normalization coefficients");
        }
        //calculate basis kernel sums
        basisKernelSums = SumsOfImages<double, KerPixel>(basisKernels);
    }

    GPU_ConvolutionImage_LC_Img(
        inImageImg,
        colPos, rowPos,
        sFn,
        sFnValGPUPtr, //output
        sFnValGPU.ptr, //output
        sfType,
        outImageImg, //output
        basisKernelsGPU.ptr,
        kernelW, kernelH,
        basisKernelSums,   //input
        normGPU_Owner.ptr,   //output
        doNormalize
    );

    //transfer input image planes to GPU
    GpuMemOwner<VarPixel> inImageGPUVar;
    inImageGPUVar.Transfer(inImageVar);
    if (inImageGPUVar.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU for input variance");
    }
    GpuMemOwner<MskPixel> inImageGPUMsk;
    inImageGPUMsk.Transfer(inImageMsk);
    if (inImageGPUMsk.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU for input mask");
    }

    // allocate output image planes on GPU
    GpuMemOwner<VarPixel > outImageGPUVar;
    outImageGPUVar.Alloc( outImageVar.Size());
    if (outImageGPUVar.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU for output variance");
    }
    GpuMemOwner<MskPixel > outImageGPUMsk;
    outImageGPUMsk.Alloc( outImageMsk.Size());
    if (outImageGPUMsk.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU for output mask");
    }
    int shMemSize = GetCudaCurSMSharedMemorySize() - shMemBytesUsed;
    int blockN = CalcBlockCount( GetCudaCurSMCount());

    cudaGetLastError(); //clear error status
    mathDetailGpu::Call_ConvolutionKernel_LC_Var(
        inImageGPUVar.ptr, inImageVar.width, inImageVar.height,
        inImageGPUMsk.ptr,
        basisKernelsGPU.ptr, kernelN,
        kernelW, kernelH,
        sFnValGPU.ptr,
        normGPU_Owner.ptr,
        outImageGPUVar.ptr,
        outImageGPUMsk.ptr,
        blockN,
        shMemSize
    );
    cudaThreadSynchronize();
    if (cudaGetLastError() != cudaSuccess)
        throw LSST_EXCEPT(GpuRuntimeError, "GPU calculation failed to run");

    CopyFromGpu(outImageVar.img, outImageGPUVar.ptr, outImageVar.Size() );
    CopyFromGpu(outImageMsk.img, outImageGPUMsk.ptr, outImageMsk.Size() );
}

#define INSTANTIATE_GPU_ConvolutionMI_LinearCombinationKernel(OutPixelT,InPixelT)  \
        template void GPU_ConvolutionMI_LinearCombinationKernel<OutPixelT,InPixelT>( \
                    GpuBuffer2D<InPixelT>& inImageImg, \
                    GpuBuffer2D<VarPixel>& inImageVar, \
                    GpuBuffer2D<MskPixel>& inImageMsk, \
                    vector<double> colPos, \
                    vector<double> rowPos, \
                    std::vector< afwMath::Kernel::SpatialFunctionPtr > sFn, \
                    GpuBuffer2D<OutPixelT>&                outImageImg, \
                    GpuBuffer2D<VarPixel>&                 outImageVar, \
                    GpuBuffer2D<MskPixel>&                 outImageMsk, \
                    std::vector< GpuBuffer2D<KerPixel> >&  basisKernels, \
                    SpatialFunctionType_t sfType, \
                    bool doNormalize  \
                    );


template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionImage_SpatiallyInvariantKernel(
    GpuBuffer2D<InPixelT>&    inImage,
    GpuBuffer2D<OutPixelT>&   outImage,
    GpuBuffer2D<KerPixel>&    kernel
)
{
    int kernelW = kernel.width;
    int kernelH = kernel.height;

    GpuMemOwner<InPixelT> inImageGPU;
    inImageGPU.Transfer(inImage);
    if (inImageGPU.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU for input image");
    }
    int shMemSize = GetCudaCurSMSharedMemorySize() - shMemBytesUsed;

    // allocate array of kernels on GPU
    GpuMemOwner<KerPixel > basisKernelGPU;
    basisKernelGPU.Transfer(kernel);
    if (basisKernelGPU.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU available for kernel");
    }
    // allocate array of output images on GPU   (one output image per kernel)
    vector< OutPixelT* > outImageGPUPtr(1);
    vector< GpuMemOwner<OutPixelT> > outImageGPU_Owner(1);

    outImageGPUPtr[0] = outImageGPU_Owner[0].Alloc( outImage.Size());
    if (outImageGPUPtr[0] == NULL) {
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU available for output image");
    }
    GpuMemOwner<OutPixelT*> outImageGPU;
    outImageGPU.TransferVec(outImageGPUPtr);

    int blockN = CalcBlockCount( GetCudaCurSMCount());

    cudaGetLastError(); //clear error status
    mathDetailGpu::Call_SpatiallyInvariantImageConvolutionKernel<OutPixelT, InPixelT>(
        inImageGPU.ptr, inImage.width, inImage.height,
        basisKernelGPU.ptr, 1,
        kernelW, kernelH,
        outImageGPU.ptr,
        blockN,
        shMemSize
    );
    cudaThreadSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        throw LSST_EXCEPT(GpuRuntimeError, "GPU calculation failed to run");
    }
    CopyFromGpu(outImage.img, outImageGPUPtr[0], outImage.Size() );
}

#define INSTANTIATE_GPU_ConvolutionImage_SpatiallyInvariantKernel(OutPixelT,InPixelT)  \
        template void GPU_ConvolutionImage_SpatiallyInvariantKernel<OutPixelT,InPixelT>( \
                    GpuBuffer2D<InPixelT>&    inImage, \
                    GpuBuffer2D<OutPixelT>&   outImage, \
                    GpuBuffer2D<KerPixel>&    kernel  \
                    );

template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionMI_SpatiallyInvariantKernel(
    GpuBuffer2D<InPixelT>&    inImageImg,
    GpuBuffer2D<VarPixel>&    inImageVar,
    GpuBuffer2D<MskPixel>&    inImageMsk,
    GpuBuffer2D<OutPixelT>&   outImageImg,
    GpuBuffer2D<VarPixel>&    outImageVar,
    GpuBuffer2D<MskPixel>&    outImageMsk,
    GpuBuffer2D<KerPixel>&    kernel
)
{
    int kernelW = kernel.width;
    int kernelH = kernel.height;

    GpuMemOwner<InPixelT> inImageGPUImg;
    inImageGPUImg.Transfer(inImageImg);
    if (inImageGPUImg.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU available for input image");
    }
    GpuMemOwner<VarPixel> inImageGPUVar;
    inImageGPUVar.Transfer(inImageVar);
    if (inImageGPUVar.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU available for input variance");
    }
    GpuMemOwner<MskPixel> inImageGPUMsk;
    inImageGPUMsk.Transfer(inImageMsk);
    if (inImageGPUMsk.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU available for input mask");
    }
    int shMemSize = GetCudaCurSMSharedMemorySize() - shMemBytesUsed;

    //allocate kernel on GPU
    GpuMemOwner<KerPixel > basisKernelGPU;
    basisKernelGPU.Transfer(kernel);
    if (basisKernelGPU.ptr == NULL)
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU available for kernel");

    // allocate array of output image planes on GPU
    vector< OutPixelT* > outImageGPUPtrImg(1);
    vector< VarPixel*  > outImageGPUPtrVar(1);
    vector< MskPixel*  > outImageGPUPtrMsk(1);

    vector< GpuMemOwner<OutPixelT> > outImageGPU_OwnerImg(1);
    vector< GpuMemOwner<VarPixel > > outImageGPU_OwnerVar(1);
    vector< GpuMemOwner<MskPixel > > outImageGPU_OwnerMsk(1);

    outImageGPUPtrImg[0] = outImageGPU_OwnerImg[0].Alloc( outImageImg.Size());
    if (outImageGPUPtrImg[0] == NULL) {
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU available for output image");
    }
    outImageGPUPtrVar[0] = outImageGPU_OwnerVar[0].Alloc( outImageVar.Size());
    if (outImageGPUPtrVar[0] == NULL) {
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU available for output variance");
    }
    outImageGPUPtrMsk[0] = outImageGPU_OwnerMsk[0].Alloc( outImageMsk.Size());
    if (outImageGPUPtrMsk[0] == NULL) {
        throw LSST_EXCEPT(GpuMemoryError, "Not enough memory on GPU available for output mask");
    }

    GpuMemOwner<OutPixelT*> outImageGPUImg;
    outImageGPUImg.TransferVec(outImageGPUPtrImg);
    GpuMemOwner<VarPixel*> outImageGPUVar;
    outImageGPUVar.TransferVec(outImageGPUPtrVar);
    GpuMemOwner<MskPixel*> outImageGPUMsk;
    outImageGPUMsk.TransferVec(outImageGPUPtrMsk);

    int blockN = CalcBlockCount( GetCudaCurSMCount());

    mathDetailGpu::Call_SpatiallyInvariantImageConvolutionKernel<OutPixelT, InPixelT>(
        inImageGPUImg.ptr, inImageImg.width, inImageImg.height,
        basisKernelGPU.ptr, 1,
        kernelW, kernelH,
        outImageGPUImg.ptr,
        blockN,
        shMemSize
    );
    //square kernel
    for (int y = 0; y < kernelH; y++) {
        for (int x = 0; x < kernelW; x++) {
            kernel.Pixel(x, y) *= kernel.Pixel(x, y);
        }
    }

    CopyFromGpu(outImageImg.img, outImageGPUPtrImg[0], outImageImg.Size() );

    basisKernelGPU.CopyToGpu(kernel);

    cudaGetLastError(); //clear last error

    mathDetailGpu::Call_SpatiallyInvariantImageConvolutionKernel<VarPixel, VarPixel>(
        inImageGPUVar.ptr, inImageVar.width, inImageVar.height,
        basisKernelGPU.ptr, 1,
        kernelW, kernelH,
        outImageGPUVar.ptr,
        blockN,
        shMemSize
    );

    cudaThreadSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        throw LSST_EXCEPT(GpuRuntimeError, "GPU variance calculation failed to run");
    }
    mathDetailGpu::Call_SpatiallyInvariantMaskConvolutionKernel(
        inImageGPUMsk.ptr, inImageMsk.width, inImageMsk.height,
        basisKernelGPU.ptr, 1,
        kernelW, kernelH,
        outImageGPUMsk.ptr,
        blockN,
        shMemSize
    );
    cudaThreadSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        throw LSST_EXCEPT(GpuRuntimeError, "GPU mask calculation failed to run");
    }
    CopyFromGpu(outImageVar.img, outImageGPUPtrVar[0], outImageVar.Size() );
    CopyFromGpu(outImageMsk.img, outImageGPUPtrMsk[0], outImageMsk.Size() );
}

#define INSTANTIATE_GPU_ConvolutionMI_SpatiallyInvariantKernel(OutPixelT,InPixelT)  \
        template void GPU_ConvolutionMI_SpatiallyInvariantKernel<OutPixelT,InPixelT>( \
                    GpuBuffer2D<InPixelT>&    inImageImg,  \
                    GpuBuffer2D<VarPixel>&    inImageVar,  \
                    GpuBuffer2D<MskPixel>&    inImageMsk,  \
                    GpuBuffer2D<OutPixelT>&   outImageImg, \
                    GpuBuffer2D<VarPixel>&    outImageVar, \
                    GpuBuffer2D<MskPixel>&    outImageMsk, \
                    GpuBuffer2D<KerPixel>&    kernel   \
                    );

/*
 * Explicit instantiation
 */
/// \cond

#define INSTANTIATE(OutPixelT,InPixelT) \
    INSTANTIATE_GPU_ConvolutionImage_LinearCombinationKernel(OutPixelT,InPixelT) \
    INSTANTIATE_GPU_ConvolutionMI_LinearCombinationKernel(OutPixelT,InPixelT) \
    INSTANTIATE_GPU_ConvolutionImage_SpatiallyInvariantKernel(OutPixelT,InPixelT) \
    INSTANTIATE_GPU_ConvolutionMI_SpatiallyInvariantKernel(OutPixelT,InPixelT)


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

#endif //GPU_BUILD



