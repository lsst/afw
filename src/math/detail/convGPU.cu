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
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

#define NVCC_COMPILING

#include <assert.h>

#include <cstdint>

#include "lsst/afw/image/LsstImageTypes.h"
#include "lsst/afw/math/detail/convCUDA.h"

namespace lsst {
namespace afw {
namespace math {
namespace detail {
namespace gpu {

typedef unsigned char uint8;
extern __shared__ uint8 smem[];

typedef KerPixel dfloat;

namespace {

__host__ __device__
int CeilDivide(int num, int divisor)    {
    return (num + divisor - 1) / divisor;
}

/**
    Loads a part of image to shared memory.
    Can handle edges of image. Data outside the edge is not copied to shared memory.

    @arg smemImg - pointer to destination in shared memory
    @arg img     - pointer to source in global memory
    @arg imgW    - width of img
    @arg imgH    - height of img
    @arg x       - x cordinate in img, top left corner of rectangle to be copied
    @arg y       - y cordinate in img, top left corner of rectangle to be copied
    @arg simgPitchX  - width of box to be copied
    @arg simgPitchY  - height of box to be copied
*/
template <typename InPixelT, typename ArithPixelT>
__device__ void LoadImageToSmem(ArithPixelT* smemImg, InPixelT* img, int imgW, int imgH,
                                int x, int y, int simgPitchX, int simgPitchY)
{
    if (x + simgPitchX <= imgW && y + simgPitchY <= imgH) {
        for (int i = 0; i < simgPitchY; i++) {
            for (int j = threadIdx.x; j < simgPitchX; j += blockDim.x) {
                smemImg[i*simgPitchX+j] = img[(i+y)*imgW+j+x];
            }
        }
    } else {
        for (int i = 0; i < simgPitchY; i++) {
            for (int j = threadIdx.x; j < simgPitchX; j += blockDim.x) {
                if ((i + y)*imgW + j + x < imgW * imgH) {
                    smemImg[i*simgPitchX+j] = img[(i+y)*imgW+j+x];
                }
            }
        }
    }
}

} //local namespace ends



/* =============================================================================
 *
 *          SPATIAL FUNCTIONS: CHEBYSHEV AND POLYNOMIAL
 *
 */

/// Calculates values of chebyshev function at location given by yCoeffs and y.
/// yCoeffs must contain values for given row
__device__ double ChebyshevLineY (
    int order,
    double y,
    double* yCoeffs
)
{
    if (order < 2)
        return yCoeffs[0] + order * yCoeffs[1] * y ;

    double CshM2 = yCoeffs[order];
    double CshM1 = 2 * y * yCoeffs[order] + yCoeffs[order-1];

    for (int i = order - 2; i > 0; i--) {
        double CshNext = 2 * y * CshM1 + yCoeffs[i] - CshM2;
        CshM2 = CshM1;
        CshM1 = CshNext;
    }

    return y * CshM1 + yCoeffs[0] - CshM2;
}

/// Calculates values of polynomial function at location given by yCoeffs and y.
/// yCoeffs must contain values for given row
__device__ double PolynomialLineY (
    int order,
    double y,
    double* yCoeffs
)
{
    double retVal = yCoeffs[order];
    for (int yCoeffInd = order - 1; yCoeffInd >= 0; --yCoeffInd) {
        retVal = (retVal * y) + yCoeffs[yCoeffInd];
    }
    return retVal;
}

/// Calculates _yCoeffs for row given in xPrime.
/// xCheby must point to memory for temporary storage
__device__ void Chebyshev_T_of_X (
    const int order,
    double xPrime,
    double* _params,
    double* _yCoeffs,  //output
    double* _xCheby    //temporary
)
{
    if (threadIdx.x >= blockSizeX) return;

    int paramN = (order + 1) * (order + 2) / 2;

    /* Solve as follows:
    - f(x,y) = Cy0 T0(y') + Cy1 T1(y') + Cy2 T2(y') + Cy3 T3(y') + ...
    where:
        Cy0 = P0 T0(x') + P1 T1(x') + P3 T2(x') + P6 T3(x') + ...
        Cy1 = P2 T0(x') + P4 T1(x') + P7 T2(x') + ...
        Cy2 = P5 T0(x') + P8 T1(x') + ...
        Cy3 = P9 T0(x') + ...
        ...

    First compute Tn(x') for each n
    Then use that to compute Cy0, Cy1, ...Cyn
    */

    // Compute _xCheby[i] = Ti(x') using the standard recurrence relationship;
    _xCheby[0] = 1;
    // note that _initialize already set _xCheby[0] = 1.
    if (order > 0) {
        _xCheby[1] = xPrime;
    }
    for (int xInd = 2; xInd <= order; ++xInd) {
        _xCheby[xInd] = (2 * xPrime * _xCheby[xInd-1]) - _xCheby[xInd-2];
    }

    // Initialize _yCoeffs to right-hand terms of equation shown in documentation block
    int paramInd = paramN - 1;
    for (int yCoeffInd = order, xChebyInd = 0; yCoeffInd >= 0;
            --yCoeffInd, ++xChebyInd, --paramInd) {
        _yCoeffs[yCoeffInd] = _params[paramInd] * _xCheby[xChebyInd];
    }

    // Add the remaining terms to _yCoeffs (starting from _order-1 because _yCoeffs[_order] is done)
    for (int startYCoeffInd = order - 1, yCoeffInd = startYCoeffInd, xChebyInd = 0;
            paramInd >= 0; --paramInd) {
        _yCoeffs[yCoeffInd] += _params[paramInd] * _xCheby[xChebyInd];
        if (yCoeffInd == 0) {
            --startYCoeffInd;
            yCoeffInd = startYCoeffInd;
            xChebyInd = 0;
        } else {
            --yCoeffInd;
            ++xChebyInd;
        }
    }

}

/// Calculates _yCoeffs for row given in xPrime.
/// xCheby must point to memory for temporary storage
__device__ void Polynomial_T_of_X (
    const int order,
    double x,
    double* _params,
    double* _yCoeffs  //output
)
{
    if (threadIdx.x >= blockSizeX) return;

    int paramN = (order + 1) * (order + 2) / 2;

    /* Solve as follows:
        - f(x,y) = Cy0 + Cy1 y + Cy2 y^2 + Cy3 y^3 + ...
        where:
       Cy0 = P0 + P1 x + P3 x^2 + P6 x^3 + ...
       Cy1 = P2 + P4 x + P7 x2 + ...
       Cy2 = P5 + P8 x + ...
       Cy3 = P9 + ...
       ...

        First compute Cy0, Cy1...Cyn by solving 1-d polynomials in x in the usual way.
        Then compute f(x,y) by solving the 1-d polynomial in y in the usual way.
        */
    const int maxYCoeffInd = order;
    int paramInd = paramN - 1;
    // initialize the y coefficients
    for (int yCoeffInd = maxYCoeffInd; yCoeffInd >= 0; --yCoeffInd, --paramInd) {
        _yCoeffs[yCoeffInd] = _params[paramInd];
    }

    // finish computing the y coefficients
    for (int startYCoeffInd = maxYCoeffInd - 1, yCoeffInd = startYCoeffInd;
            paramInd >= 0; --paramInd) {
        _yCoeffs[yCoeffInd] = (_yCoeffs[yCoeffInd] * x) + _params[paramInd];
        if (yCoeffInd == 0) {
            --startYCoeffInd;
            yCoeffInd = startYCoeffInd;
        } else {
            --yCoeffInd;
        }
    }

}

__global__ void ChebyshevImageValues(
    double* out, int outW, int outH,
    int order,
    double* params,
    double* rowPos,
    double* colPos,
    double minX, double minY, double maxX, double maxY
)
{
    const double scaleX = 2.0 / (maxX - minX);
    const double scaleY = 2.0 / (maxY - minY);
    const double offsetX = -(minX + maxX) * 0.5;
    const double offsetY = -(minY + maxY) * 0.5;

    const int coeffN = order + 1;
    double* smemDbl = (double*)smem;

    const int coeffPadding = coeffN + 1 - (coeffN % 2);

    double* yCoeffsAll = smemDbl ;
    double* xChebyAll = yCoeffsAll + (coeffPadding * blockSizeX);
    double* smemParams = xChebyAll + (coeffPadding * blockSizeX);

    int yLineGroup = threadIdx.x % blockSizeX;

    double* yCoeffs = yCoeffsAll + (coeffPadding * yLineGroup);
    double* xCheby = xChebyAll  + (coeffPadding * yLineGroup);

    //load params to shared memory
    int paramN = (order + 1) * (order + 2) / 2;
    for(int i = threadIdx.x; i < paramN; i += blockDim.x)
        smemParams[i] = params[i];

    int totalBlocks = CeilDivide(outW, blockSizeX);
    int totalPixelsInBlock = blockSizeX * outH;

    for (int blkI = blockIdx.x; blkI < totalBlocks; blkI += gridDim.x)
    {
        int blkX = blkI * blockSizeX;
        int blkY = 0;

        int curPixelX = threadIdx.x % blockSizeX;
        int curPixelY = threadIdx.x / blockSizeX;

        int outPixelX = blkX + curPixelX;
        const int x = colPos[outPixelX];
        const double xPrime = (x + offsetX) * scaleX;

        __syncthreads();
        Chebyshev_T_of_X(order, xPrime, smemParams, yCoeffs, xCheby);
        __syncthreads();

        for(int curPixel = threadIdx.x; curPixel < totalPixelsInBlock; curPixel += blockDim.x)
        {
            int outPixelY = blkY + curPixelY;

            if (outPixelX >= outW || outPixelY >= outH) continue;

            const int y = rowPos[outPixelY];
            const double yPrime = (y + offsetY) * scaleY;

            out[outPixelY*outW + outPixelX] = ChebyshevLineY(order, yPrime, yCoeffs);

            curPixelY += blockDim.x / blockSizeX;
        }
    }

}

__global__ void PolynomialImageValues(
    double* out, int outW, int outH,
    int order,
    double* params,
    double* rowPos,
    double* colPos
)
{
    const int coeffN = order + 1;
    double* smemDbl = (double*)smem;

    const int coeffPadding = coeffN + 1 - (coeffN % 2);

    double* yCoeffsAll = smemDbl ;
    double* smemParams = yCoeffsAll + (coeffPadding * blockSizeX);

    int yLineGroup = threadIdx.x % blockSizeX;

    double* yCoeffs = yCoeffsAll + (coeffPadding * yLineGroup);

    //load params to shared memory
    int paramN = (order + 1) * (order + 2) / 2;
    for(int i = threadIdx.x; i < paramN; i += blockDim.x)
        smemParams[i] = params[i];

    int totalBlocks = CeilDivide(outW, blockSizeX);
    int totalPixelsInBlock = blockSizeX * outH;

    for (int blkI = blockIdx.x; blkI < totalBlocks; blkI += gridDim.x)
    {
        int blkX = blkI * blockSizeX;
        int blkY = 0;

        int curPixelX = threadIdx.x % blockSizeX;
        int curPixelY = threadIdx.x / blockSizeX;

        int outPixelX = blkX + curPixelX;
        const int x = colPos[outPixelX];

        __syncthreads();
        Polynomial_T_of_X(order, x, smemParams, yCoeffs);
        __syncthreads();

        for(int curPixel = threadIdx.x; curPixel < totalPixelsInBlock; curPixel += blockDim.x)
        {
            int outPixelY = blkY + curPixelY;

            if (outPixelX >= outW || outPixelY >= outH) continue;

            const int y = rowPos[outPixelY];

            out[outPixelY*outW + outPixelX] = PolynomialLineY(order, y, yCoeffs);

            curPixelY += blockDim.x / blockSizeX;
        }
    }

}

/// calculates values of Chebyshev function for every pixel in the image
void Call_ChebyshevImageValues(
    double* out, int outW, int outH,
    int order,
    double* params,
    double* rowPos,
    double* colPos,
    double minX, double minY, double maxX, double maxY,
    int sharedMemorySize
)
{
    dim3 block(256);
    dim3 grid(CeilDivide(outW, blockSizeX));

    ChebyshevImageValues <<< grid, block, sharedMemorySize >>>(
        out, outW, outH, order, params,
        rowPos, colPos,
        minX, minY, maxX, maxY
    );
}

/// calculates values of polynomial function for every pixel in the image
void Call_PolynomialImageValues(
    double* out, int outW, int outH,
    int order,
    double* params,
    double* rowPos,
    double* colPos,
    int sharedMemorySize
)
{
    dim3 block(256);
    dim3 grid(CeilDivide(outW, blockSizeX));

    PolynomialImageValues <<< grid, block, sharedMemorySize >>>(
        out, outW, outH, order, params,
        rowPos, colPos
    );
}

__global__ void NormalizationImageValues(
    double* out, int outW, int outH,
    double** sFn, int n,
    double* kernelSum, bool* isDivideByZeroGPU
)
{
    double*  smemDbl = (double*)smem;
    double*  smemKernelSum = smemDbl ;
    double** smemSfnPtr = (double**) (smemKernelSum + n);

    //load kernel sums into shared memory
    for(int i = threadIdx.x; i < n; i += blockDim.x) {
        smemKernelSum[i] = kernelSum[i];
    }
    //load pointers to sFn values into shared memory
    for(int i = threadIdx.x; i < n; i += blockDim.x) {
        smemSfnPtr[i] = sFn[i];
    }
    __syncthreads();

    int totalPixels = outW * outH;

    for(int curPixel = threadIdx.x; curPixel < totalPixels; curPixel += blockDim.x)
    {
        //int outPixelX=curPixel%outW;
        //int outPixelY=curPixel/outW;

        double sum = 0;
        for (int i = 0; i < n; i++)  {
            sum += smemSfnPtr[i][curPixel] * smemKernelSum[i];
        }
        if (sum == 0)  *isDivideByZeroGPU = true;
        else         out[curPixel] = 1.0 / sum;
    }

}

/// calculates normalization values for every pixel in the image
void Call_NormalizationImageValues(
    double* out, int outW, int outH,
    double** sFn, int n,
    double* kernelSum,
    bool* isDivideByZeroGPU,
    int blockN,
    int sharedMemorySize
)
{
    dim3 block(256);
    dim3 grid(blockN);

    NormalizationImageValues <<< grid, block, sharedMemorySize >>>(
        out, outW, outH,
        sFn, n,
        kernelSum,
        isDivideByZeroGPU
    );
}


/* =============================================================================
 *
 *          MULTIPLE SPATIALLY INVARIANT KERNELS
 *            (single input image, multiple kernels, multiple output images)
 *
 *          USED FOR:
 *              - spatially invariant kernel (image and variance planes)
 *              - linear combination kernel  (image plane)
 *
 */

//#define SumUpPixelProduct(n)  sum+=pixLine[n] * filtLine[n];

#define SumUpPixelProduct(n)  if (filtLine[n]!=0) sum+=pixLine[n] * filtLine[n];

#define SumUpPixelProductX4(n)      \
            SumUpPixelProduct(n)   \
            SumUpPixelProduct(n+1) \
            SumUpPixelProduct(n+2) \
            SumUpPixelProduct(n+3)

#if 0  //simpler but slower version of ApplyFilterOnce (without unrolling)

/**
    Convolves filter in smemfilt with part of image loadad at start of shared memory.
    Convolves only one pixel, given by curPixelX and curPixelY, of image in shared memory.
*/
__device__ dfloat ApplyFilterOnce(
    dfloat* smemFilt, int filtW, int filtH,
    int curPixelX, int curPixelY, int simgPitchX
)
{
    dfloat* smemImg = (dfloat*)smem;
    dfloat totalSum = 0;
    dfloat* pixLine = &smemImg[curPixelY*simgPitchX+curPixelX];
    dfloat* filtLine = smemFilt;
    int pixLineAdd = simgPitchX - filtW;

    for (int filtY = 0; filtY < filtH; filtY++) {
        dfloat sum = 0;
#pragma unroll 4
        for (int x = 0; x < filtW; x++) {
            if (*filtLine != 0) {
                sum += *pixLine * *filtLine;
            }
            pixLine++;
            filtLine++;
        }
        pixLine += pixLineAdd;
        totalSum += sum;
    }
    return totalSum;
}

#else //unrolled version of ApplyFilterOnce

/**
    Convolves filter in smemfilt with part of image loadad at start of shared memory.
    Convolves only one pixel, given by curPixelX and curPixelY, of image in shared memory.
*/
__device__ dfloat ApplyFilterOnce(
    dfloat* smemFilt, int filtW, int filtH,
    int curPixelX, int curPixelY, int simgPitchX
)
{
    dfloat* smemImg = (dfloat*)smem;
    dfloat totalSum = 0;
    dfloat* pixLineOrig = &smemImg[curPixelY*simgPitchX+curPixelX];
    dfloat* filtLineOrig = smemFilt;
    int remainingFiltW = filtW;
    int pixLineAdd = simgPitchX;
    int procWidth;

    if (remainingFiltW >= 12) {
        procWidth = 24;
        while (remainingFiltW >= procWidth) {
            dfloat* pixLine = pixLineOrig;
            dfloat* filtLine = filtLineOrig;
            for (int filtY = 0; filtY < filtH; filtY++) {
                dfloat sum = 0;
                SumUpPixelProductX4(0)
                SumUpPixelProductX4(4)
                SumUpPixelProductX4(8)
                SumUpPixelProductX4(12)
                SumUpPixelProductX4(16)
                SumUpPixelProductX4(20)

                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum += sum;
            }
            remainingFiltW -= procWidth;
            pixLineOrig += procWidth;
            filtLineOrig += procWidth;
        }

        procWidth = 12;
        if (remainingFiltW >= procWidth) {
            dfloat* pixLine = pixLineOrig;
            dfloat* filtLine = filtLineOrig;
            for (int filtY = 0; filtY < filtH; filtY++) {
                dfloat sum = 0;
                SumUpPixelProductX4(0)
                SumUpPixelProductX4(4)
                SumUpPixelProductX4(8)

                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum += sum;
            }
            remainingFiltW -= procWidth;
            pixLineOrig += procWidth;
            filtLineOrig += procWidth;
        }

        if (remainingFiltW == 0) return totalSum;
    }

    dfloat* pixLine = pixLineOrig;
    dfloat* filtLine = filtLineOrig;

    if (remainingFiltW < 4) {
        if (remainingFiltW == 1)
            for (int filtY = 0; filtY < filtH; filtY++) {
                dfloat sum = 0;
                SumUpPixelProduct(0)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum += sum;
            }
        else if (remainingFiltW == 2)
            for (int filtY = 0; filtY < filtH; filtY++) {
                dfloat sum = 0;
                SumUpPixelProduct(0)
                SumUpPixelProduct(1)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum += sum;
            }
        else if (remainingFiltW == 3)
            for (int filtY = 0; filtY < filtH; filtY++) {
                dfloat sum = 0;
                SumUpPixelProduct(0)
                SumUpPixelProduct(1)
                SumUpPixelProduct(2)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum += sum;
            }
        return totalSum;
    }
    if (remainingFiltW < 9) {
        if (remainingFiltW == 4) {
            for (int filtY = 0; filtY < filtH; filtY++) {
                dfloat sum = 0;
                SumUpPixelProductX4(0)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum += sum;
            }
        } else if (remainingFiltW == 5) {
            for (int filtY = 0; filtY < filtH; filtY++) {
                dfloat sum = 0;
                SumUpPixelProductX4(0)
                SumUpPixelProduct(4)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum += sum;
            }
        } else if (remainingFiltW == 6) {
            for (int filtY = 0; filtY < filtH; filtY++) {
                dfloat sum = 0;
                SumUpPixelProductX4(0)
                SumUpPixelProduct(4)
                SumUpPixelProduct(5)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum += sum;
            }
        } else if (remainingFiltW == 7) {
            for (int filtY = 0; filtY < filtH; filtY++) {
                dfloat sum = 0;
                SumUpPixelProductX4(0)
                SumUpPixelProduct(4)
                SumUpPixelProduct(5)
                SumUpPixelProduct(6)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum += sum;
            }
        } else if (remainingFiltW == 8) {
            for (int filtY = 0; filtY < filtH; filtY++) {
                dfloat sum = 0;
                SumUpPixelProductX4(0)
                SumUpPixelProductX4(4)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum += sum;
            }
        }
        return totalSum;
    }
    if (remainingFiltW == 9) {
        for (int filtY = 0; filtY < filtH; filtY++) {
            dfloat sum = 0;
            SumUpPixelProductX4(0)
            SumUpPixelProductX4(4)
            SumUpPixelProduct(8)
            filtLine += filtW;
            pixLine += pixLineAdd;
            totalSum += sum;
        }
    } else if (remainingFiltW == 10) {
        for (int filtY = 0; filtY < filtH; filtY++) {
            dfloat sum = 0;
            SumUpPixelProductX4(0)
            SumUpPixelProductX4(4)
            SumUpPixelProduct(8)
            SumUpPixelProduct(9)
            filtLine += filtW;
            pixLine += pixLineAdd;
            totalSum += sum;
        }
    } else if (remainingFiltW == 11) {
        for (int filtY = 0; filtY < filtH; filtY++) {
            dfloat sum = 0;
            SumUpPixelProductX4(0)
            SumUpPixelProductX4(4)
            SumUpPixelProduct(8)
            SumUpPixelProduct(9)
            SumUpPixelProduct(10)
            filtLine += filtW;
            pixLine += pixLineAdd;
            totalSum += sum;
        }
    }
    return totalSum;
}

#endif

template <typename OutPixelT, typename InPixelT>
__global__ void SpatiallyInvariantImgConvolutionKernel(
    InPixelT* img, int imgW, int imgH,
    dfloat* allFilt, int filtN,
    int filtW, int filtH,
    OutPixelT** result
)
{
    const int outW = imgW - filtW + 1;
    const int outH = imgH - filtH + 1;

    int simgPitchX = blockSizeX + filtW - 1;
    int simgPitchY = blockSizeY + filtH - 1;

    /*int simgSize=simgPitchX*simgPitchY;
    dfloat* smemFiltBeg=(dfloat*)smem + simgSize;

    for (int filtI=filtStart; filtI<filtStart+filtN; filtI++) {
        dfloat* smemFilt=smemFiltBeg + (filtW*filtH)*(filtI-filtStart);
        for(int i=threadIdx.x; i<filtW*filtH; i+=blockDim.x)
            smemFilt[i]=filt[filtI][i];
        }*/

    int blockNX = CeilDivide(outW, blockSizeX);
    int blockNY = CeilDivide(outH, blockSizeY);

    int totalBlocks = blockNX * blockNY;
    int totalPixelsInBlock = blockSizeX * blockSizeY;

    for (int blkI = blockIdx.x; blkI < totalBlocks; blkI += gridDim.x)
    {
        int blkX = blkI % blockNX;
        int blkY = blkI / blockNX;

        int x = blkX * blockSizeX;
        int y = blkY * blockSizeY;
        __syncthreads();
        LoadImageToSmem((dfloat*) smem, img, imgW, imgH, x, y, simgPitchX, simgPitchY);
        __syncthreads();

        int curPixelX = threadIdx.x % blockSizeX;
        int curPixelY = threadIdx.x / blockSizeX;

        for(int curPixel = threadIdx.x; curPixel < totalPixelsInBlock; curPixel += blockDim.x)
        {
            int outPixelX = x + curPixelX;
            int outPixelY = y + curPixelY;

            if (outPixelX >= outW || outPixelY >= outH) continue;

            for (int filtI = 0; filtI < filtN; filtI++) {
                dfloat* filtPtr = &allFilt[filtI*filtW*filtH];
                //dfloat* smemFilt=smemFiltBeg + (filtW*filtH)*(filtI-filtStart);

                //dfloat sum = ApplyFilterOnce(smemFilt, filtW, filtH, curPixelX, curPixelY, simgPitchX);
                dfloat sum = ApplyFilterOnce(filtPtr, filtW, filtH, curPixelX, curPixelY, simgPitchX);

                OutPixelT* curResultImg = result[filtI];
                curResultImg[outPixelY*outW + outPixelX] = OutPixelT(sum);
            }

            curPixelX += blockDim.x;
            while (curPixelX >= blockSizeX) {
                curPixelX -= blockSizeX;
                curPixelY++;
            }
        }
    }

}

template <typename OutPixelT, typename InPixelT>
void Call_SpatiallyInvariantImageConvolutionKernel(
    InPixelT*  inImageGPU, int inImageWidth, int inImageHeight,
    KerPixel*  allKernelsGPU, int kernelTotalN,
    int kernelW, int kernelH,
    OutPixelT* outImageGPU[],
    int blockN,
    int sharedMemorySize
)
{
    dim3 block(256);
    dim3 grid(blockN);

    SpatiallyInvariantImgConvolutionKernel<OutPixelT, InPixelT> <<< grid, block, sharedMemorySize >>>(
        inImageGPU, inImageWidth, inImageHeight,
        allKernelsGPU, kernelTotalN,
        kernelW, kernelH,
        outImageGPU
    );
}

#define INSTANTIATE_SpatiallyInvariantImageConvolutionKernel(OutPixelT,InPixelT)  \
template void Call_SpatiallyInvariantImageConvolutionKernel<OutPixelT,InPixelT>( \
        InPixelT*  inImageGPU, int inImageWidth, int inImageHeight, \
        KerPixel*  allKernelsGPU, int kernelTotalN,  \
        int kernelW, int kernelH,         \
        OutPixelT* outImageGPU[], \
        int blockN, \
        int sharedMemorySize    \
        );

/* =============================================================================
 *
 *          LINEAR COMBINATION KERNEL - IMAGE
 *
 */

namespace {

/**
    Convolves calculated filter with part of image loadad at start of the shared memory.
    Convolves only one pixel, given by curPixelX and curPixelY, of image in shared memory.

    Input image block should be placed at start of shared memory

    @arg allFilt - pointer to all filters (placed sequantially)
    @arg filtN   - number of filters
    @arg sfval   - values of spatial functions for each filter at current pixel
    @arg normval - normalization coefficient
    @arg simgPitchX - size of both part of image in shared memory and part of image mask data

    @return result of convolving image data
*/
__device__ dfloat ApplyFilterOnceLCImg(
    dfloat* allFilt, int filtN, int filtW, int filtH,
    double  sfVal[],
    double normVal,
    int curPixelX, int curPixelY, int simgPitchX
)
{
    dfloat* smemImg = (dfloat*)smem;
    dfloat totalSum = 0;
    dfloat* pixLine = &smemImg[curPixelY*simgPitchX+curPixelX];
    int pixLineAdd = simgPitchX - filtW;
    int kernelSize =  filtW * filtH;

    int filtRemainder = filtN % 3;

    for (int y = 0; y < filtH; y++) {
        dfloat sum = 0;
        for (int x = 0; x < filtW; x++) {
            dfloat* filtLine = allFilt + y * filtW + x;

            dfloat filtVal;

            filtVal = *filtLine * sfVal[0];
            filtLine += kernelSize;

            int filtI = 1;
            if (filtRemainder == 2) {
                filtVal += *filtLine * sfVal[1];
                filtLine += kernelSize;
                filtI = 2;
            } else if (filtRemainder == 0) {
                filtVal += *filtLine * sfVal[1];
                filtLine += kernelSize;
                filtVal += *filtLine * sfVal[2];
                filtLine += kernelSize;
                filtI = 3;
            }

            while(filtI < filtN) {
                filtVal += *filtLine * sfVal[filtI];
                filtLine += kernelSize;
                filtI++;
                filtVal += *filtLine * sfVal[filtI];
                filtLine += kernelSize;
                filtI++;
                filtVal += *filtLine * sfVal[filtI];
                filtLine += kernelSize;
                filtI++;
            }

            filtVal *= normVal;
            if (filtVal != 0) {
                sum += *pixLine * filtVal;
            }
            pixLine++;
        }
        pixLine += pixLineAdd;
        totalSum += sum;
    }
    return totalSum;
}

} //local namespace ends

template <typename OutPixelT, typename InPixelT>
__global__ void ConvolutionKernel_LC_Img(
    InPixelT* img, int imgW, int imgH,
    dfloat* filt, int filtN,
    int filtW, int filtH,
    double**  sfValImg,
    double* norm,
    OutPixelT* out
)
{
    //Asserts that : blockDim.x is divisible by blockSizeX

    const int outW = imgW - filtW + 1;
    const int outH = imgH - filtH + 1;

    int simgPitchX = blockSizeX + filtW - 1;
    int simgPitchY = blockSizeY + filtH - 1;

    /*int simgSize=simgPitchX*simgPitchY;

    dfloat* smemFiltBeg=(dfloat*)smem + simgSize;

    for (int filtI=filtStart; filtI<filtStart+filtN; filtI++) {
        dfloat* smemFilt=smemFiltBeg + (filtW*filtH)*(filtI-filtStart);
        for(int i=threadIdx.x; i<filtW*filtH; i+=blockDim.x)
            smemFilt[i]=filt[filtI][i];
        }*/

    int blockNX = CeilDivide(outW, blockSizeX);
    int blockNY = CeilDivide(outH, blockSizeY);

    int totalBlocks = blockNX * blockNY;
    int totalPixelsInBlock = blockSizeX * blockSizeY;

    for (int blkI = blockIdx.x; blkI < totalBlocks; blkI += gridDim.x)
    {
        int blkX = blkI % blockNX;
        int blkY = blkI / blockNX;

        int x = blkX * blockSizeX;
        int y = blkY * blockSizeY;

        __syncthreads();
        LoadImageToSmem((dfloat*) smem, img, imgW, imgH, x, y, simgPitchX, simgPitchY);
        __syncthreads();

        int curPixelX = threadIdx.x % blockSizeX;
        int curPixelY = threadIdx.x / blockSizeX;

        for(int curPixel = threadIdx.x; curPixel < totalPixelsInBlock; curPixel += blockDim.x)
        {
            int outPixelX = x + curPixelX;
            int outPixelY = y + curPixelY;
            int outAddr = outPixelY * outW + outPixelX;

            if (outPixelX >= outW || outPixelY >= outH) continue;

            double sfVal[maxGpuSfCount];
            for (int filtI = 0; filtI < filtN; filtI++)
                sfVal[filtI] = sfValImg[filtI][outAddr];

            double normVal = 1;
            if (norm != NULL)
                normVal = norm[outAddr];

            double sum = ApplyFilterOnceLCImg(filt, filtN, filtW, filtH,
                                              sfVal, normVal, curPixelX, curPixelY, simgPitchX);
            out[outAddr] = OutPixelT(sum);

            curPixelY += blockDim.x / blockSizeX;
        }
    }

}

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
)
{
    dim3 block(256);
    dim3 grid(blockN);

    ConvolutionKernel_LC_Img <<< grid, block, sharedMemorySize >>>(
        inImageGPU, inImageWidth, inImageHeight,
        kernelGPU, kernelTotalN,
        kernelW, kernelH,
        sfValGPU,
        normGPU,
        outImageGPU
    );
}

#define INSTANTIATE_ConvolutionKernel_LC_Img(OutPixelT,InPixelT)  \
template void Call_ConvolutionKernel_LC_Img<OutPixelT,InPixelT>( \
        InPixelT*  inImageGPU, int inImageWidth, int inImageHeight, \
        KerPixel*  kernelGPU, int kernelTotalN, \
        int kernelW, int kernelH, \
        double* sfValGPU[], \
        double* normGPU, \
        OutPixelT* outImageGPU, \
        int blockN, \
        int sharedMemorySize \
        );

/* =============================================================================
 *
 *          LINEAR COMBINATION KERNEL - VARIANCE AND MASK
 *
 */

namespace {

/**
    Convolves calculated filter with part of image loadad at start of shared memory.
    Convolves only one pixel, given by curPixelX and curPixelY, of image in shared memory.

    Input variance should be placed at start of shared memory

    @arg smemMsk - pointer to input image mask data
    @arg mskSum  - output parameter, result of convolving mask data
    @arg allFilt - pointer to all filters (placed sequantially)
    @arg filtN   - number of filters
    @arg sfval   - values of spatial functions for each filter at current pixel
    @arg normval - normalization coefficient
    @arg simgPitchX - size of both part of image in shared memory and part of image mask data

    @return result of convolving variance data
*/
__device__ dfloat ApplyFilterOnceLCVar(
    MskPixel* smemMsk,
    MskPixel& mskSum,
    dfloat* allFilt, int filtN, int filtW, int filtH,
    double  sfVal[],
    double normVal,
    int curPixelX, int curPixelY, int simgPitchX
)
{
    dfloat* smemImg = (dfloat*)smem;
    dfloat totalSum = 0;
    dfloat* pixLine = &smemImg[curPixelY*simgPitchX+curPixelX];
    MskPixel* pixMskLine = &smemMsk[curPixelY*simgPitchX+curPixelX];
    int pixLineAdd = simgPitchX - filtW;
    int kernelSize =  filtW * filtH;

    mskSum = 0;
    int filtRemainder = filtN % 3;

    for (int y = 0; y < filtH; y++) {
        dfloat sum = 0;
        for (int x = 0; x < filtW; x++) {
            dfloat* filtLine = allFilt + y * filtW + x;

            dfloat filtVal = *filtLine * sfVal[0];
            filtLine += kernelSize;

            int filtI = 1;
            if (filtRemainder == 2) {
                filtVal += *filtLine * sfVal[1];
                filtLine += kernelSize;
                filtI = 2;
            } else if (filtRemainder == 0) {
                filtVal += *filtLine * sfVal[1];
                filtLine += kernelSize;
                filtVal += *filtLine * sfVal[2];
                filtLine += kernelSize;
                filtI = 3;
            }

            while(filtI < filtN) {
                filtVal += *filtLine * sfVal[filtI];
                filtLine += kernelSize;
                filtI++;
                filtVal += *filtLine * sfVal[filtI];
                filtLine += kernelSize;
                filtI++;
                filtVal += *filtLine * sfVal[filtI];
                filtLine += kernelSize;
                filtI++;
            }

            filtVal *= normVal;
            if (filtVal != 0) {
                sum += *pixLine * (filtVal * filtVal);
                mskSum |= *pixMskLine;
            }
            pixLine++;
            pixMskLine++;
        }
        pixLine += pixLineAdd;
        pixMskLine += pixLineAdd;
        totalSum += sum;
    }
    return totalSum;
}

} //local namespace ends

__global__ void ConvolutionKernel_LC_Var(
    VarPixel* img, int imgW, int imgH,
    MskPixel*  inMsk,
    dfloat* filt, int filtN,
    int filtW, int filtH,
    double**  sfValImg,
    double* norm,
    VarPixel* outVar,
    MskPixel* outMsk
)
{
    //Asserts that : blockDim.x is divisible by blockSizeX

    const int outW = imgW - filtW + 1;
    const int outH = imgH - filtH + 1;

    int simgPitchX = blockSizeX + filtW - 1;
    int simgPitchY = blockSizeY + filtH - 1;

    int simgSize = simgPitchX * simgPitchY;
    MskPixel* smemMsk = (MskPixel*)((dfloat*)smem + simgSize);

    /*dfloat* smemFiltBeg=(dfloat*)smem + simgSize;

    for (int filtI=filtStart; filtI<filtStart+filtN; filtI++) {
        dfloat* smemFilt=smemFiltBeg + (filtW*filtH)*(filtI-filtStart);
        for(int i=threadIdx.x; i<filtW*filtH; i+=blockDim.x)
            smemFilt[i]=filt[filtI][i];
        }*/

    int blockNX = CeilDivide(outW, blockSizeX);
    int blockNY = CeilDivide(outH, blockSizeY);

    int totalBlocks = blockNX * blockNY;
    int totalPixelsInBlock = blockSizeX * blockSizeY;

    for (int blkI = blockIdx.x; blkI < totalBlocks; blkI += gridDim.x)
    {
        int blkX = blkI % blockNX;
        int blkY = blkI / blockNX;

        int x = blkX * blockSizeX;
        int y = blkY * blockSizeY;

        __syncthreads();
        LoadImageToSmem((dfloat*) smem, img, imgW, imgH, x, y, simgPitchX, simgPitchY);
        LoadImageToSmem(        smemMsk, inMsk, imgW, imgH, x, y, simgPitchX, simgPitchY);
        __syncthreads();

        int curPixelX = threadIdx.x % blockSizeX;
        int curPixelY = threadIdx.x / blockSizeX;

        for(int curPixel = threadIdx.x; curPixel < totalPixelsInBlock; curPixel += blockDim.x)
        {
            int outPixelX = x + curPixelX;
            int outPixelY = y + curPixelY;
            int outAddr = outPixelY * outW + outPixelX;

            if (outPixelX >= outW || outPixelY >= outH) continue;

            double sfVal[maxGpuSfCount];
            for (int filtI = 0; filtI < filtN; filtI++) {
                sfVal[filtI] = sfValImg[filtI][outAddr];
            }

            double normVal = 1;
            if (norm != NULL) normVal = norm[outAddr];

            MskPixel mskSum;
            dfloat sum = ApplyFilterOnceLCVar(smemMsk, mskSum, filt, filtN, filtW, filtH,
                                              sfVal, normVal, curPixelX, curPixelY, simgPitchX);

            outVar[outAddr] = sum;
            outMsk[outAddr] = mskSum;

            curPixelY += blockDim.x / blockSizeX;
        }
    }

}

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
)
{
    dim3 block(256);
    dim3 grid(blockN);

    ConvolutionKernel_LC_Var <<< grid, block, sharedMemorySize >>>(
        inImageGPU, inImageWidth, inImageHeight,
        inMskGPU,
        kernelGPU, kernelTotalN,
        kernelW, kernelH,
        sfValGPU,
        normGPU,
        outImageGPU,
        outMskGPU
    );
}

/* =============================================================================
 *
 *          SPATIALLY INVARIANT KERNEL - MASK PLANE
 *
 */

namespace {

//#define SumUpPixelProductMask(n)  sum |=pixLine[n];

#define SumUpPixelProductMask(n)  if (filtLine[n]!=0) sum |=pixLine[n] ;

#define SumUpPixelProductMaskX4(n)      \
            SumUpPixelProductMask(n)   \
            SumUpPixelProductMask(n+1) \
            SumUpPixelProductMask(n+2) \
            SumUpPixelProductMask(n+3)

#if 0  //simpler but slower version of MaskApplyFilterOnce (without unrolling)

/**
    Convolves filter in smemfilt with part of image mask loadad at start of shared memory.
    Convolves only one pixel, given by curPixelX and curPixelY, of image mask in shared memory.
*/
__device__ MskPixel MaskApplyFilterOnce(
    dfloat* smemFilt, int filtW, int filtH,
    int curPixelX, int curPixelY, int simgPitchX
)
{
    MskPixel* smemImg = (MskPixel*)smem;
    MskPixel totalSum = 0;
    MskPixel* pixLine = &smemImg[curPixelY*simgPitchX+curPixelX];
    dfloat* filtLine = smemFilt;
    int pixLineAdd = simgPitchX - filtW;

    for (int filtY = 0; filtY < filtH; filtY++) {
        MskPixel sum = 0;
#pragma unroll 4
        for (int x = 0; x < filtW; x++) {
            if (*filtLine != 0) {
                sum |= *pixLine;
            }
            pixLine++;
            filtLine++;
        }
        pixLine += pixLineAdd;
        totalSum |= sum;
    }
    return totalSum;
}

#else //unrolled version of MaskApplyFilterOnce

/**
    Convolves filter in smemfilt with part of image mask loadad at start of shared memory.
    Convolves only one pixel, given by curPixelX and curPixelY, of image mask in shared memory.
*/
__device__ MskPixel MaskApplyFilterOnce(
    dfloat* smemFilt, int filtW, int filtH,
    int curPixelX, int curPixelY, int simgPitchX
)
{
    MskPixel* smemImg = (MskPixel*)smem;
    MskPixel totalSum = 0;
    MskPixel* pixLineOrig = &smemImg[curPixelY*simgPitchX+curPixelX];
    dfloat* filtLineOrig = smemFilt;
    int remainingFiltW = filtW;
    int pixLineAdd = simgPitchX;
    int procWidth;

    if (remainingFiltW >= 12) {
        procWidth = 24;
        while (remainingFiltW >= procWidth) {
            MskPixel* pixLine = pixLineOrig;
            dfloat* filtLine = filtLineOrig;
            for (int filtY = 0; filtY < filtH; filtY++) {
                MskPixel sum = 0;
                SumUpPixelProductMaskX4(0)
                SumUpPixelProductMaskX4(4)
                SumUpPixelProductMaskX4(8)
                SumUpPixelProductMaskX4(12)
                SumUpPixelProductMaskX4(16)
                SumUpPixelProductMaskX4(20)

                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum |= sum;
            }
            remainingFiltW -= procWidth;
            pixLineOrig += procWidth;
            filtLineOrig += procWidth;
        }

        procWidth = 12;
        if (remainingFiltW >= procWidth) {
            MskPixel* pixLine = pixLineOrig;
            dfloat* filtLine = filtLineOrig;
            for (int filtY = 0; filtY < filtH; filtY++) {
                MskPixel sum = 0;
                SumUpPixelProductMaskX4(0)
                SumUpPixelProductMaskX4(4)
                SumUpPixelProductMaskX4(8)

                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum |= sum;
            }
            remainingFiltW -= procWidth;
            pixLineOrig += procWidth;
            filtLineOrig += procWidth;
        }

        if (remainingFiltW == 0) return totalSum;
    }

    MskPixel* pixLine = pixLineOrig;
    dfloat* filtLine = filtLineOrig;

    if (remainingFiltW < 4) {
        if (remainingFiltW == 1) {
            for (int filtY = 0; filtY < filtH; filtY++) {
                MskPixel sum = 0;
                SumUpPixelProductMask(0)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum |= sum;
            }
        } else if (remainingFiltW == 2) {
            for (int filtY = 0; filtY < filtH; filtY++) {
                MskPixel sum = 0;
                SumUpPixelProductMask(0)
                SumUpPixelProductMask(1)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum |= sum;
            }
        } else if (remainingFiltW == 3) {
            for (int filtY = 0; filtY < filtH; filtY++) {
                MskPixel sum = 0;
                SumUpPixelProductMask(0)
                SumUpPixelProductMask(1)
                SumUpPixelProductMask(2)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum |= sum;
            }
        }
        return totalSum;
    }
    if (remainingFiltW < 9) {
        if (remainingFiltW == 4) {
            for (int filtY = 0; filtY < filtH; filtY++) {
                MskPixel sum = 0;
                SumUpPixelProductMaskX4(0)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum |= sum;
            }
        } else if (remainingFiltW == 5) {
            for (int filtY = 0; filtY < filtH; filtY++) {
                MskPixel sum = 0;
                SumUpPixelProductMaskX4(0)
                SumUpPixelProductMask(4)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum |= sum;
            }
        } else if (remainingFiltW == 6) {
            for (int filtY = 0; filtY < filtH; filtY++) {
                MskPixel sum = 0;
                SumUpPixelProductMaskX4(0)
                SumUpPixelProductMask(4)
                SumUpPixelProductMask(5)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum |= sum;
            }
        } else if (remainingFiltW == 7) {
            for (int filtY = 0; filtY < filtH; filtY++) {
                MskPixel sum = 0;
                SumUpPixelProductMaskX4(0)
                SumUpPixelProductMask(4)
                SumUpPixelProductMask(5)
                SumUpPixelProductMask(6)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum |= sum;
            }
        } else if (remainingFiltW == 8) {
            for (int filtY = 0; filtY < filtH; filtY++) {
                MskPixel sum = 0;
                SumUpPixelProductMaskX4(0)
                SumUpPixelProductMaskX4(4)
                filtLine += filtW;
                pixLine += pixLineAdd;
                totalSum |= sum;
            }
        }
        return totalSum;
    }
    if (remainingFiltW == 9) {
        for (int filtY = 0; filtY < filtH; filtY++) {
            MskPixel sum = 0;
            SumUpPixelProductMaskX4(0)
            SumUpPixelProductMaskX4(4)
            SumUpPixelProductMask(8)
            filtLine += filtW;
            pixLine += pixLineAdd;
            totalSum |= sum;
        }
    } else if (remainingFiltW == 10) {
        for (int filtY = 0; filtY < filtH; filtY++) {
            MskPixel sum = 0;
            SumUpPixelProductMaskX4(0)
            SumUpPixelProductMaskX4(4)
            SumUpPixelProductMask(8)
            SumUpPixelProductMask(9)
            filtLine += filtW;
            pixLine += pixLineAdd;
            totalSum |= sum;
        }
    } else if (remainingFiltW == 11) {
        for (int filtY = 0; filtY < filtH; filtY++) {
            MskPixel sum = 0;
            SumUpPixelProductMaskX4(0)
            SumUpPixelProductMaskX4(4)
            SumUpPixelProductMask(8)
            SumUpPixelProductMask(9)
            SumUpPixelProductMask(10)
            filtLine += filtW;
            pixLine += pixLineAdd;
            totalSum |= sum;
        }
    }
    return totalSum;
}

#endif

} //local namespace ends

__global__ void SpatiallyInvariantMaskConvolutionKernel(
    MskPixel* img, int imgW, int imgH,
    dfloat* allFilt, int filtN,
    int filtW, int filtH,
    MskPixel** result
)
{
    const int outW = imgW - filtW + 1;
    const int outH = imgH - filtH + 1;

    int simgPitchX = blockSizeX + filtW - 1;
    int simgPitchY = blockSizeY + filtH - 1;


    /*int simgSize=simgPitchX*simgPitchY;
    dfloat* smemFiltBeg=(dfloat*)smem + simgSize;

    for (int filtI=filtStart; filtI<filtStart+filtN; filtI++) {
        dfloat* smemFilt=smemFiltBeg + (filtW*filtH)*(filtI-filtStart);
        for(int i=threadIdx.x; i<filtW*filtH; i+=blockDim.x)
            smemFilt[i]=filt[filtI][i];
        }*/

    int blockNX = CeilDivide(outW, blockSizeX);
    int blockNY = CeilDivide(outH, blockSizeY);

    int totalBlocks = blockNX * blockNY;
    int totalPixelsInBlock = blockSizeX * blockSizeY;

    for (int blkI = blockIdx.x; blkI < totalBlocks; blkI += gridDim.x)
    {
        int blkX = blkI % blockNX;
        int blkY = blkI / blockNX;

        int x = blkX * blockSizeX;
        int y = blkY * blockSizeY;
        __syncthreads();
        LoadImageToSmem((MskPixel*) smem, img, imgW, imgH, x, y, simgPitchX, simgPitchY);
        __syncthreads();

        int curPixelX = threadIdx.x % blockSizeX;
        int curPixelY = threadIdx.x / blockSizeX;

        for(int curPixel = threadIdx.x; curPixel < totalPixelsInBlock; curPixel += blockDim.x)
        {
            int outPixelX = x + curPixelX;
            int outPixelY = y + curPixelY;

            if (outPixelX >= outW || outPixelY >= outH) continue;

            for (int filtI = 0; filtI < filtN; filtI++) {
                //dfloat* smemFilt=smemFiltBeg + (filtW*filtH)*(filtI-filtStart);
                dfloat* filtPtr = &allFilt[filtI*filtW*filtH];

                MskPixel sum = MaskApplyFilterOnce(filtPtr, filtW, filtH, curPixelX, curPixelY, simgPitchX);

                MskPixel* curResultImg = result[filtI];
                curResultImg[outPixelY*outW + outPixelX] = sum;
            }

            curPixelX += blockDim.x;
            while (curPixelX >= blockSizeX) {
                curPixelX -= blockSizeX;
                curPixelY++;
            }
        }
    }

}

void Call_SpatiallyInvariantMaskConvolutionKernel(
    MskPixel*  inImageGPU, int inImageWidth, int inImageHeight,
    KerPixel*  allKernelsGPU, int kernelTotalN,
    int kernelW, int kernelH,
    MskPixel* outImageGPU[],
    int blockN,
    int sharedMemorySize
)
{
    dim3 block(256);
    dim3 grid(blockN);

    SpatiallyInvariantMaskConvolutionKernel <<< grid, block, sharedMemorySize >>>(
        inImageGPU, inImageWidth, inImageHeight,
        allKernelsGPU, kernelTotalN,
        kernelW, kernelH,
        outImageGPU
    );
}


#define INSTANTIATE(OutPixelT,InPixelT) \
    INSTANTIATE_SpatiallyInvariantImageConvolutionKernel(OutPixelT,InPixelT) \
    INSTANTIATE_ConvolutionKernel_LC_Img(OutPixelT,InPixelT)

/*
 * Explicit instantiation
 */
/// \cond

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


// ================== GPU kernel for testing ======================

template <typename T>
__global__ void Test(T* ret)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId == 0) ret[0] = 5;
    if (threadId == 1) ret[1] = 8;

}

template <typename T>
void CallTestGpuKernel(T* ret)
{
    dim3 block(192);
    dim3 grid(60);

    Test <<< grid, block>>>(ret);

}


template void CallTestGpuKernel<int>(int*);

}
}
}
}
} //namespace lsst::afw::math::detail::gpu ends



