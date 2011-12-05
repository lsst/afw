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


#ifndef GPU_BUILD //build this file only if requested

#include <stdio.h>
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/detail/Convolve.h"

namespace lsst {
namespace afw {
namespace math {
namespace detail {
namespace gpu {

void PrintCudaDeviceInfo() {
    printf("Afw not compiled with GPU support\n");
}

int GetCudaCurDeviceId() {
    throw LSST_EXCEPT(GpuRuntimeErrorException, "AFW not built with GPU support");
}

int GetCudaCurSMSharedMemorySize(){
    throw LSST_EXCEPT(GpuRuntimeErrorException, "AFW not built with GPU support");
}

int GetCudaCurGlobalMemorySize(){
    throw LSST_EXCEPT(GpuRuntimeErrorException, "AFW not built with GPU support");
}

int GetCudaCurSMRegisterCount(){
    throw LSST_EXCEPT(GpuRuntimeErrorException, "AFW not built with GPU support");
}

int GetCudaCurSMCount(){
    throw LSST_EXCEPT(GpuRuntimeErrorException, "AFW not built with GPU support");
}

bool GetCudaCurIsDoublePrecisionSupported(){
    throw LSST_EXCEPT(GpuRuntimeErrorException, "AFW not built with GPU support");
}

void SetCudaDevice(int devId){
    throw LSST_EXCEPT(GpuRuntimeErrorException, "AFW not built with GPU support");
}

void CudaReserveDevice(){
    throw LSST_EXCEPT(GpuRuntimeErrorException, "AFW not built with GPU support");
}

void CudaThreadExit(){
    throw LSST_EXCEPT(GpuRuntimeErrorException, "AFW not built with GPU support");
}

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
bool SelectPreferredCudaDevice()
{
    throw LSST_EXCEPT(GpuRuntimeErrorException, "AFW not built with gpu support");
}
void AutoSelectCudaDevice()
{
    throw LSST_EXCEPT(GpuRuntimeErrorException, "AFW not built with gpu support");
}
void VerifyCudaDevice()
{
    throw LSST_EXCEPT(GpuRuntimeErrorException, "AFW not built with gpu support");
}



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

#include "lsst/afw/math/detail/ImageBuffer.h"
#include "lsst/afw/math/detail/cudaConvWrapper.h"
#include "lsst/afw/math/detail/cudaQueryDevice.h"
#include "lsst/afw/math/detail/Convolve.h"

using namespace std;
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

namespace gpu {

void PrintDeviceProperties(int id, cudaDeviceProp deviceProp)
{
    printf("Name : %s  |", deviceProp.name );
    printf("  CUDA Capable SM %d.%d hardware, %d multiproc.\n", deviceProp.major, deviceProp.minor,
           deviceProp.multiProcessorCount);
    printf("   Clock rate:       %6.2f GHz \t", deviceProp.clockRate / (1000.0 * 1000));
    printf("   Memory on device: %6zu MiB\n", deviceProp.totalGlobalMem / (1 << 20) );
    printf("   Multiprocessors:  %6d\n", deviceProp.multiProcessorCount);
    printf("       Warp size:    %6d \t",  deviceProp.warpSize );
    printf("       Shared memory:%6zu KiB\n", deviceProp.sharedMemPerBlock / (1 << 10) );
    printf("       Registers:    %6d \t", deviceProp.regsPerBlock );
    printf("       Max threads:  %6d \n", deviceProp.maxThreadsPerBlock );

    printf("   Compute mode (device sharing) : ");
    if (deviceProp.computeMode == cudaComputeModeDefault) {
        printf("Default - shared between threads\n" );
    }
    if (deviceProp.computeMode == cudaComputeModeExclusive) {
        printf("Exclusive - only one thread at a time\n" );
    }
    if (deviceProp.computeMode == cudaComputeModeProhibited) {
        printf("Prohibited - cannot use this device\n" );
    }

    printf("   Timeout enabled: %3s  ", deviceProp.kernelExecTimeoutEnabled == 1 ? "Yes" : "No" );
    printf("   Overlapped copying: %3s  ", deviceProp.deviceOverlap == 1 ? "Yes" : "No" );
    printf("   Intergrated on MB: %3s\n", deviceProp.integrated == 1 ? "Yes" : "No" );
    printf("   Memory pitch: %12zu \t", deviceProp.memPitch );
    printf("   Constant memory: %6zu kiB \n", deviceProp.totalConstMem / (1 << 10) );
}

void PrintCudaErrorInfo(cudaError_t cudaError, const char* errorStr)
{
    printf("\nSupplied error string: %s\n", errorStr);
    printf(  "CUDA error           : %d\n", cudaError);
    printf(  "CUDA error string    : %s\n", cudaGetErrorString(cudaError));
    exit(0);
}

void PrintCudaDeviceInfo()
{
    fflush(stdout);

    cudaError_t cudaError;

    int driverVersion;
    cudaError = cudaDriverGetVersion(&driverVersion);
    if (cudaError != cudaSuccess) PrintCudaErrorInfo(cudaError, "Could not get CUDA driver version");
    printf("Driver ver.: %d.%d   ", driverVersion / 1000, driverVersion % 1000);
    fflush(stdout);

    int runtimeVersion;
    cudaError = cudaRuntimeGetVersion(&runtimeVersion);
    if (cudaError != cudaSuccess) PrintCudaErrorInfo(cudaError, "Could not get CUDA runtime version");
    printf("Runtime ver.: %d.%d   ", runtimeVersion / 1000, runtimeVersion % 1000);
    fflush(stdout);

    //int preferredDeviceId = 0;

    int cudaDevicesN = 0;
    cudaError = cudaGetDeviceCount(&cudaDevicesN);
    if (cudaError != cudaSuccess) PrintCudaErrorInfo(cudaError, "Could not get CUDA device count");

    printf("Device count: %d   ", cudaDevicesN);
    fflush(stdout);
    if(cudaDevicesN < 1) {
        printf("Your system does not have a CUDA capable device\n");
        exit(0);
    }

    int curDevId;
    cudaError = cudaGetDevice(&curDevId);
    if (cudaError != cudaSuccess) PrintCudaErrorInfo(cudaError, "Could not get CUDA device id");
    printf("Info for device %d\n", curDevId);
    fflush(stdout);

    cudaDeviceProp deviceProp;
    cudaError = cudaGetDeviceProperties(&deviceProp, curDevId);
    if (cudaError != cudaSuccess) PrintCudaErrorInfo(cudaError, "Could not get CUDA device properties");
    PrintDeviceProperties(curDevId, deviceProp);
    fflush(stdout);

    for (int i = 0; i < 79; i++) {
        printf("-");
    }
    printf("\n");
    fflush(stdout);
}

int GetPreferredCudaDevice()
{
    const char *devStr = getenv("CUDA_DEVICE");
    if (devStr == NULL) return -2;
    else                return atoi(devStr);
}

bool SelectPreferredCudaDevice()
{
    int devId = GetPreferredCudaDevice();

    //printf("DEVICE ID %d\n", devId);

    if (devId >= 0) {
        cudaError_t err = cudaSetDevice(devId);
        if (err != cudaSuccess) {
            cudaGetLastError(); //clear error code
            char errorStr[1000];
            sprintf(errorStr, "Error selecting device %d:\n %s\n", devId, cudaGetErrorString(err));
            throw LSST_EXCEPT(GpuRuntimeErrorException, errorStr);
        }
        return true;
    }

    if (devId != -2) return true;

    return false;
}

cudaDeviceProp GetDesiredDeviceProperties()
{
    cudaDeviceProp prop;
    memset(&prop, 1, sizeof(prop));

    //min sm 1.3
    prop.major = 1;
    prop.minor = 3;

    prop.maxGridSize[0] = 128;
    prop.maxThreadsDim[0] = 256;

    prop.multiProcessorCount = 2;
    prop.clockRate = 700.0 * 1000 ; // 700 MHz
    prop.warpSize = 32 ;
    prop.sharedMemPerBlock = 32 * (1 << 10); //32 KiB
    prop.regsPerBlock = 256 * 60 ;
    prop.maxThreadsPerBlock = 256;
    prop.totalGlobalMem = 500 * 1024 * 1024;

    return prop;
}

void AutoSelectCudaDevice()
{
    int cudaDevicesN = 0;
    cudaGetDeviceCount(&cudaDevicesN);
    if (cudaDevicesN == 0) {
        throw LSST_EXCEPT(GpuRuntimeErrorException, "No CUDA capable GPUs found");
    }

    cudaDeviceProp prop = GetDesiredDeviceProperties();
    char errorStr[1000];

    int devId;
    cudaError_t cudaError = cudaChooseDevice(&devId, &prop);
    //printf("Error device %d:\n %s\n", devId, cudaGetErrorString(err));
    if (cudaError != cudaSuccess) {
        throw LSST_EXCEPT(GpuRuntimeErrorException, "Error choosing device automatically");
    }
    cudaError = cudaSetDevice(devId);
    if (cudaError == cudaErrorSetOnActiveProcess) {
        cudaGetDevice(&devId);
    } else if (cudaError != cudaSuccess) {
        cudaGetLastError(); //clear error
        sprintf(errorStr, "Error automatically selecting device %d:\n %s\n",
                devId, cudaGetErrorString(cudaError));
        throw LSST_EXCEPT(GpuRuntimeErrorException, errorStr);
    }
}

void VerifyCudaDevice()
{
    cudaDeviceProp prop = GetDesiredDeviceProperties();
    char errorStr[1000];

    int devId;
    cudaError_t cudaError = cudaGetDevice(&devId);
    if (cudaError != cudaSuccess) {
        throw LSST_EXCEPT(GpuRuntimeErrorException, "Could not get selected CUDA device ID");
    }
    cudaDeviceProp deviceProp;
    cudaError = cudaGetDeviceProperties(&deviceProp, devId);
    if (cudaError != cudaSuccess) {
        throw LSST_EXCEPT(GpuRuntimeErrorException, "Could not get CUDA device properties");
    }
    if (deviceProp.major < prop.major ||
            (deviceProp.major == prop.major && deviceProp.minor < prop.minor)
       ) {
        sprintf(errorStr, "Only SM %d.%d or better GPU devices are currently allowed", prop.major, prop.minor);
        throw LSST_EXCEPT(GpuRuntimeErrorException, errorStr );
    }

    if (deviceProp.major == prop.major && deviceProp.minor < prop.minor) {
        if (deviceProp.totalGlobalMem < prop.totalGlobalMem) {
            throw LSST_EXCEPT(GpuRuntimeErrorException, "Not enough global memory on GPU");
        }
    }
    if (deviceProp.sharedMemPerBlock < 16 * 1000) {
        throw LSST_EXCEPT(GpuRuntimeErrorException, "Not enough shared memory on GPU");
    }
    if (deviceProp.regsPerBlock < prop.regsPerBlock) {
        throw LSST_EXCEPT(GpuRuntimeErrorException, "Not enough registers per block available on GPU");
    }
    if (deviceProp.maxThreadsPerBlock < prop.maxThreadsPerBlock) {
        throw LSST_EXCEPT(GpuRuntimeErrorException, "Not enough threads per block available on GPU");
    }
}

int GetCudaCurDeviceId()
{
    int curDevId;
    cudaError_t cudaError = cudaGetDevice(&curDevId);
    if (cudaError != cudaSuccess) PrintCudaErrorInfo(cudaError, "GetCudaDeviceId> Could not get CUDA device id");
    return curDevId;
}

int GetCudaCurSMSharedMemorySize()
{
    int curDevId = GetCudaCurDeviceId();
    cudaDeviceProp deviceProp;
    cudaError_t cudaError = cudaGetDeviceProperties(&deviceProp, curDevId);
    if (cudaError != cudaSuccess) PrintCudaErrorInfo(cudaError, "GetCudaSMSharedMemorySize> Could not get CUDA device properties");

    return deviceProp.sharedMemPerBlock;
}

int GetCudaCurGlobalMemorySize()
{
    int curDevId = GetCudaCurDeviceId();
    cudaDeviceProp deviceProp;
    cudaError_t cudaError = cudaGetDeviceProperties(&deviceProp, curDevId);
    if (cudaError != cudaSuccess) {
        PrintCudaErrorInfo(cudaError, "GetCudaCurGlobalMemorySize> Could not get CUDA device properties");
    }
    return deviceProp.totalGlobalMem;
}

int GetCudaCurSMRegisterCount()
{
    int curDevId = GetCudaCurDeviceId();
    cudaDeviceProp deviceProp;
    cudaError_t cudaError = cudaGetDeviceProperties(&deviceProp, curDevId);
    if (cudaError != cudaSuccess) {
        PrintCudaErrorInfo(cudaError, "GetCudaSMRegisterCount> Could not get CUDA device properties");
    }
    return deviceProp.regsPerBlock;
}

int GetCudaCurSMCount()
{
    int curDevId = GetCudaCurDeviceId();
    cudaDeviceProp deviceProp;
    cudaError_t cudaError = cudaGetDeviceProperties(&deviceProp, curDevId);
    if (cudaError != cudaSuccess) {
        PrintCudaErrorInfo(cudaError, "GetCudaSMCount> Could not get CUDA device properties");
    }
    return deviceProp.multiProcessorCount;
}

bool GetCudaCurIsDoublePrecisionSupported()
{
    int curDevId = GetCudaCurDeviceId();
    cudaDeviceProp deviceProp;
    cudaError_t cudaError = cudaGetDeviceProperties(&deviceProp, curDevId);
    if (cudaError != cudaSuccess) {
        PrintCudaErrorInfo(cudaError, "GetCudaIsDoublePrecisionSupported> Could not get CUDA device properties");
    }
    return deviceProp.major >= 2 || (deviceProp.major == 1 && deviceProp.minor >= 3);
}

void SetCudaDevice(int devId)
{
    cudaError_t cudaError = cudaSetDevice(devId);
    if (cudaError != cudaSuccess) PrintCudaErrorInfo(cudaError, "SetCudaDevice> unsucessfull");
}

void CudaReserveDevice()
{
    int* dataGpu;
    cudaError_t cudaError = cudaMalloc((void**)&dataGpu, 256 * sizeof(int));
    if (cudaError != cudaSuccess) {
        PrintCudaErrorInfo(cudaError, "CudaReserveDevice> Could not reserve device by calling cudaMalloc");
    }
    cudaError = cudaFree(dataGpu);
    if (cudaError != cudaSuccess) {
        PrintCudaErrorInfo(cudaError, "CudaReserveDevice> Could not release memory by calling cudaFree");
    }
}

void CudaThreadExit()
{
    cudaThreadExit();
}

template<typename T>
T* AllocOnGpu(int size)
{
    T* dataGpu;
    cudaError_t cudaError = cudaMalloc((void**)&dataGpu, size * sizeof(T));
    if (cudaError != cudaSuccess) {
        return NULL;
    }
    return dataGpu;
}
template<typename T>
void CopyFromGpu(T* destCpu, T* sourceGpu, int size)
{
    cudaError_t cudaError = cudaMemcpy(
                                /* Desination:*/     destCpu,
                                /* Source:    */     sourceGpu,
                                /* Size in bytes: */ size * sizeof(T),
                                /* Direction   */    cudaMemcpyDeviceToHost
                            );
    if (cudaError != cudaSuccess)
        throw LSST_EXCEPT(GpuMemoryException, "CopyFromGpu: failed");
}
template<typename T>
void CopyToGpu(T* destGpu, T* sourceCpu, int size)
{
    cudaError_t cudaError;
    cudaError = cudaMemcpy(
                    /* Desination:*/     destGpu,
                    /* Source:    */     sourceCpu,
                    /* Size in bytes: */ size * sizeof(T),
                    /* Direction   */    cudaMemcpyHostToDevice
                );
    if (cudaError != cudaSuccess) {
        throw LSST_EXCEPT(GpuMemoryException, "CopyToGpu: failed");
    }
}

template<typename T>
T* TransferToGpu(const T* sourceCpu, int size)
{
    T* dataGpu;
    cudaError_t cudaError = cudaMalloc((void**)&dataGpu, size * sizeof(T));
    if (cudaError != cudaSuccess) {
        return NULL;
    }
    cudaError = cudaMemcpy(
                    /* Desination:*/     dataGpu,
                    /* Source:    */     sourceCpu,
                    /* Size in bytes: */ size * sizeof(T),
                    /* Direction   */    cudaMemcpyHostToDevice
                );
    if (cudaError != cudaSuccess) {
        throw LSST_EXCEPT(GpuMemoryException, "TransferToGpu: transfer failed");
    }
    return dataGpu;
}

/**
    A class for handling GPU memory managment and copying data to and from GPU

    Automatically releases GPU memory on destruction, simplifying GPU memory management
*/
template<typename T>
class GpuMemOwner
{
public:
    T* ptr;
    int size;
    GpuMemOwner() : ptr(NULL) {}

    T* Transfer(const T* source, int size_p) {
        assert(ptr == NULL);
        size = size_p;
        ptr = TransferToGpu(source, size);
        return ptr;
    }
    T* Transfer(const ImageBuffer<T>& source) {
        assert(ptr == NULL);
        size = source.Size();
        ptr = TransferToGpu(source.img, size);
        return ptr;
    }
    T* TransferVec(const vector<T>& source) {
        assert(ptr == NULL);
        size = int(source.size());
        ptr = TransferToGpu(&source[0], size);
        return ptr;
    }
    T* Alloc(int size_p)  {
        assert(ptr == NULL);
        size = size_p;
        ptr = AllocOnGpu<T>(size);
        return ptr;
    }
    T* CopyToGpu(ImageBuffer<T>& source) {
        assert(ptr != NULL);
        assert(source.Size() == size);
        gpu::CopyToGpu(ptr, source.img, size);
        return ptr;
    }
    T* CopyFromGpu(ImageBuffer<T>& dest) {
        assert(ptr != NULL);
        assert(dest.Size() == size);
        gpu::CopyFromGpu(dest.img, ptr, size);
        return ptr;
    }

    ~GpuMemOwner() {
        if (ptr != NULL) cudaFree(ptr);
    }
};

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

    int* resGpu = gpu::AllocOnGpu<int>(2);

    CallTestGpuKernel(resGpu);

    gpu::CopyFromGpu(res, resGpu, 2);

    ret1 = res[0];
    ret2 = res[1];
}

} // namespace lsst::afw::math::detail::gpu ends

namespace {

//calculates sum of each image in 'images' vector
template <typename ResultT, typename InT>
vector<ResultT> SumsOfImages(const vector< ImageBuffer<InT> >&  images)
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
    const ImageBuffer<InPixelT>& inImage,
    const vector<double>& colPos,
    const vector<double>& rowPos,
    const std::vector< afwMath::Kernel::SpatialFunctionPtr >& sFn,
    const vector<double*>& sFnValGPUPtr, //output
    double** sFnValGPU,    //output
    SpatialFunctionType_t sfType,
    ImageBuffer<OutPixelT>&  outImage, //output
    KerPixel*   basisKernelsListGPU,
    int kernelW, int kernelH,
    const vector<double>&   basisKernelSums,   //input
    double* normGPU,   //output
    bool doNormalize
)
{
    const int kernelN = sFn.size();

    //transfer input image
    gpu::GpuMemOwner<InPixelT > inImageGPU;
    inImageGPU.Transfer(inImage);
    if (inImageGPU.ptr == NULL)  {
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU for input image");
    }
    // allocate output image planes on GPU
    gpu::GpuMemOwner<OutPixelT> outImageGPU;
    outImageGPU.Alloc( outImage.Size());
    if (outImageGPU.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU for output image");
    }
    //transfer coordinate tranform data
    gpu::GpuMemOwner<double> colPosGPU_Owner;
    colPosGPU_Owner.TransferVec(colPos);
    if (colPosGPU_Owner.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryException,
                          "Not enough memory on GPU for row coordinate tranformation data");
    }
    gpu::GpuMemOwner<double> rowPosGPU_Owner;
    rowPosGPU_Owner.TransferVec(rowPos);
    if (rowPosGPU_Owner.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryException,
                          "Not enough memory on GPU for column coordinate tranformation data");
    }
    vector< double* >                     sFnParamsGPUPtr(kernelN);
    vector< gpu::GpuMemOwner<double> >    sFnParamsGPU_Owner(kernelN);

    //transfer sfn parameters to GPU
    for (int i = 0; i < kernelN; i++) {
        std::vector<double> spatialParams = sFn[i]->getParameters();
        sFnParamsGPUPtr[i] = sFnParamsGPU_Owner[i].TransferVec(spatialParams);
        if (sFnParamsGPUPtr[i] == NULL) {
            throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU for spatial function parameters");
        }
    }

    int shMemSize = gpu::GetCudaCurSMSharedMemorySize() - shMemBytesUsed;

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
                throw LSST_EXCEPT(GpuRuntimeErrorException, "GPU calculation failed to run");
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
                throw LSST_EXCEPT(GpuRuntimeErrorException, "GPU calculation failed to run");
            }
        }
    }
    cudaThreadSynchronize();
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        throw LSST_EXCEPT(GpuRuntimeErrorException, "GPU calculation failed to run");
    }

    int blockN = gpu::CalcBlockCount( gpu::GetCudaCurSMCount());

    //transfer basis kernel sums
    if (doNormalize) {
        gpu::GpuMemOwner<double> basisKernelSumsGPU;
        basisKernelSumsGPU.TransferVec(basisKernelSums);

        bool isDivideByZero = false;
        gpu::GpuMemOwner<bool> isDivideByZeroGPU;
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
            throw LSST_EXCEPT(GpuRuntimeErrorException, "GPU calculation failed to run");
        }
        gpu::CopyFromGpu<bool>(&isDivideByZero, isDivideByZeroGPU.ptr, 1);
        if (isDivideByZero) {
            throw LSST_EXCEPT(pexExcept::OverflowErrorException, "Cannot normalize; kernel sum is 0");
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
        throw LSST_EXCEPT(GpuRuntimeErrorException, "GPU calculation failed to run");
    }

    gpu::CopyFromGpu(outImage.img, outImageGPU.ptr, outImage.Size() );

}

} //local namespace ends

template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionImage_LinearCombinationKernel(
    ImageBuffer<InPixelT>& inImage,
    vector<double> colPos,
    vector<double> rowPos,
    std::vector< afwMath::Kernel::SpatialFunctionPtr > sFn,
    ImageBuffer<OutPixelT>&                outImage,
    std::vector< ImageBuffer<KerPixel> >&  basisKernels,
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
    gpu::GpuMemOwner<KerPixel > basisKernelsGPU;
    basisKernelsGPU.Alloc(kernelSize * kernelN);

    for (int i = 0; i < kernelN; i++) {
        KerPixel* kernelBeg = basisKernelsGPU.ptr + (kernelSize * i);
        gpu::CopyToGpu(kernelBeg,
                       basisKernels[i].img,
                       kernelSize
                      );
    }

    // allocate array of spatial function value images on GPU
    vector< double* >                     sFnValGPUPtr(kernelN);
    vector< gpu::GpuMemOwner<double > >   sFnValGPU_Owner(kernelN);

    for (int i = 0; i < kernelN; i++) {
        sFnValGPUPtr[i] = sFnValGPU_Owner[i].Alloc(outWidth * outHeight);
        if (sFnValGPUPtr[i] == NULL) {
            throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU for spatial function values");
        }
    }
    gpu::GpuMemOwner<double*> sFnValGPU;
    sFnValGPU.TransferVec(sFnValGPUPtr);

    gpu::GpuMemOwner<double > normGPU_Owner;
    vector<double> basisKernelSums(kernelN);
    if (doNormalize) {
        //allocate normalization coeficients
        normGPU_Owner.Alloc(outWidth * outHeight);
        if (normGPU_Owner.ptr == NULL) {
            throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU for normalization coefficients");
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
                    ImageBuffer<InPixelT>& inImage, \
                    vector<double> colPos, \
                    vector<double> rowPos, \
                    std::vector< afwMath::Kernel::SpatialFunctionPtr > sFn, \
                    ImageBuffer<OutPixelT>&                outImage, \
                    std::vector< ImageBuffer<KerPixel> >&  basisKernels, \
                    SpatialFunctionType_t sfType, \
                    bool doNormalize \
                    );

template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionMI_LinearCombinationKernel(
    ImageBuffer<InPixelT>& inImageImg,
    ImageBuffer<VarPixel>& inImageVar,
    ImageBuffer<MskPixel>& inImageMsk,
    vector<double> colPos,
    vector<double> rowPos,
    std::vector< afwMath::Kernel::SpatialFunctionPtr > sFn,
    ImageBuffer<OutPixelT>&                outImageImg,
    ImageBuffer<VarPixel>&                 outImageVar,
    ImageBuffer<MskPixel>&                 outImageMsk,
    std::vector< ImageBuffer<KerPixel> >&  basisKernels,
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
    gpu::GpuMemOwner<KerPixel > basisKernelsGPU;
    basisKernelsGPU.Alloc(kernelSize * kernelN);

    for (int i = 0; i < kernelN; i++) {
        KerPixel* kernelBeg = basisKernelsGPU.ptr + (kernelSize * i);
        gpu::CopyToGpu(kernelBeg,
                       basisKernels[i].img,
                       kernelSize
                      );
    }

    //alloc sFn images on GPU
    vector< double* >                     sFnValGPUPtr(kernelN);
    vector< gpu::GpuMemOwner<double > >   sFnValGPU_Owner(kernelN);

    for (int i = 0; i < kernelN; i++) {
        sFnValGPUPtr[i] = sFnValGPU_Owner[i].Alloc(outWidth * outHeight);
        if (sFnValGPUPtr[i] == NULL) {
            throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU for spatial function values");
        }
    }
    gpu::GpuMemOwner<double*> sFnValGPU;
    sFnValGPU.TransferVec(sFnValGPUPtr);

    //allocate normalization coeficients image on GPU
    gpu::GpuMemOwner<double > normGPU_Owner;
    std::vector<KerPixel> basisKernelSums(kernelN);
    if (doNormalize) {
        //allocate normalization coeficients
        normGPU_Owner.Alloc(outWidth * outHeight);
        if (normGPU_Owner.ptr == NULL) {
            throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU for normalization coefficients");
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
    gpu::GpuMemOwner<VarPixel> inImageGPUVar;
    inImageGPUVar.Transfer(inImageVar);
    if (inImageGPUVar.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU for input variance");
    }
    gpu::GpuMemOwner<MskPixel> inImageGPUMsk;
    inImageGPUMsk.Transfer(inImageMsk);
    if (inImageGPUMsk.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU for input mask");
    }

    // allocate output image planes on GPU
    gpu::GpuMemOwner<VarPixel > outImageGPUVar;
    outImageGPUVar.Alloc( outImageVar.Size());
    if (outImageGPUVar.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU for output variance");
    }
    gpu::GpuMemOwner<MskPixel > outImageGPUMsk;
    outImageGPUMsk.Alloc( outImageMsk.Size());
    if (outImageGPUMsk.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU for output mask");
    }
    int shMemSize = gpu::GetCudaCurSMSharedMemorySize() - shMemBytesUsed;
    int blockN = gpu::CalcBlockCount( gpu::GetCudaCurSMCount());

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
        throw LSST_EXCEPT(GpuRuntimeErrorException, "GPU calculation failed to run");

    gpu::CopyFromGpu(outImageVar.img, outImageGPUVar.ptr, outImageVar.Size() );
    gpu::CopyFromGpu(outImageMsk.img, outImageGPUMsk.ptr, outImageMsk.Size() );
}

#define INSTANTIATE_GPU_ConvolutionMI_LinearCombinationKernel(OutPixelT,InPixelT)  \
        template void GPU_ConvolutionMI_LinearCombinationKernel<OutPixelT,InPixelT>( \
                    ImageBuffer<InPixelT>& inImageImg, \
                    ImageBuffer<VarPixel>& inImageVar, \
                    ImageBuffer<MskPixel>& inImageMsk, \
                    vector<double> colPos, \
                    vector<double> rowPos, \
                    std::vector< afwMath::Kernel::SpatialFunctionPtr > sFn, \
                    ImageBuffer<OutPixelT>&                outImageImg, \
                    ImageBuffer<VarPixel>&                 outImageVar, \
                    ImageBuffer<MskPixel>&                 outImageMsk, \
                    std::vector< ImageBuffer<KerPixel> >&  basisKernels, \
                    SpatialFunctionType_t sfType, \
                    bool doNormalize  \
                    );


template <typename OutPixelT, typename InPixelT>
void GPU_ConvolutionImage_SpatiallyInvariantKernel(
    ImageBuffer<InPixelT>&    inImage,
    ImageBuffer<OutPixelT>&   outImage,
    ImageBuffer<KerPixel>&    kernel
)
{
    int kernelW = kernel.width;
    int kernelH = kernel.height;

    gpu::GpuMemOwner<InPixelT> inImageGPU;
    inImageGPU.Transfer(inImage);
    if (inImageGPU.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU for input image");
    }
    int shMemSize = gpu::GetCudaCurSMSharedMemorySize() - shMemBytesUsed;

    // allocate array of kernels on GPU
    gpu::GpuMemOwner<KerPixel > basisKernelGPU;
    basisKernelGPU.Transfer(kernel);
    if (basisKernelGPU.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU available for kernel");
    }
    // allocate array of output images on GPU   (one output image per kernel)
    vector< OutPixelT* > outImageGPUPtr(1);
    vector< gpu::GpuMemOwner<OutPixelT> > outImageGPU_Owner(1);

    outImageGPUPtr[0] = outImageGPU_Owner[0].Alloc( outImage.Size());
    if (outImageGPUPtr[0] == NULL) {
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU available for output image");
    }
    gpu::GpuMemOwner<OutPixelT*> outImageGPU;
    outImageGPU.TransferVec(outImageGPUPtr);

    int blockN = gpu::CalcBlockCount( gpu::GetCudaCurSMCount());

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
        throw LSST_EXCEPT(GpuRuntimeErrorException, "GPU calculation failed to run");
    }
    gpu::CopyFromGpu(outImage.img, outImageGPUPtr[0], outImage.Size() );
}

#define INSTANTIATE_GPU_ConvolutionImage_SpatiallyInvariantKernel(OutPixelT,InPixelT)  \
        template void GPU_ConvolutionImage_SpatiallyInvariantKernel<OutPixelT,InPixelT>( \
                    ImageBuffer<InPixelT>&    inImage, \
                    ImageBuffer<OutPixelT>&   outImage, \
                    ImageBuffer<KerPixel>&    kernel  \
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
)
{
    int kernelW = kernel.width;
    int kernelH = kernel.height;

    gpu::GpuMemOwner<InPixelT> inImageGPUImg;
    inImageGPUImg.Transfer(inImageImg);
    if (inImageGPUImg.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU available for input image");
    }
    gpu::GpuMemOwner<VarPixel> inImageGPUVar;
    inImageGPUVar.Transfer(inImageVar);
    if (inImageGPUVar.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU available for input variance");
    }
    gpu::GpuMemOwner<MskPixel> inImageGPUMsk;
    inImageGPUMsk.Transfer(inImageMsk);
    if (inImageGPUMsk.ptr == NULL) {
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU available for input mask");
    }
    int shMemSize = gpu::GetCudaCurSMSharedMemorySize() - shMemBytesUsed;

    //allocate kernel on GPU
    gpu::GpuMemOwner<KerPixel > basisKernelGPU;
    basisKernelGPU.Transfer(kernel);
    if (basisKernelGPU.ptr == NULL)
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU available for kernel");

    // allocate array of output image planes on GPU
    vector< OutPixelT* > outImageGPUPtrImg(1);
    vector< VarPixel*  > outImageGPUPtrVar(1);
    vector< MskPixel*  > outImageGPUPtrMsk(1);

    vector< gpu::GpuMemOwner<OutPixelT> > outImageGPU_OwnerImg(1);
    vector< gpu::GpuMemOwner<VarPixel > > outImageGPU_OwnerVar(1);
    vector< gpu::GpuMemOwner<MskPixel > > outImageGPU_OwnerMsk(1);

    outImageGPUPtrImg[0] = outImageGPU_OwnerImg[0].Alloc( outImageImg.Size());
    if (outImageGPUPtrImg[0] == NULL) {
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU available for output image");
    }
    outImageGPUPtrVar[0] = outImageGPU_OwnerVar[0].Alloc( outImageVar.Size());
    if (outImageGPUPtrVar[0] == NULL) {
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU available for output variance");
    }
    outImageGPUPtrMsk[0] = outImageGPU_OwnerMsk[0].Alloc( outImageMsk.Size());
    if (outImageGPUPtrMsk[0] == NULL) {
        throw LSST_EXCEPT(GpuMemoryException, "Not enough memory on GPU available for output mask");
    }

    gpu::GpuMemOwner<OutPixelT*> outImageGPUImg;
    outImageGPUImg.TransferVec(outImageGPUPtrImg);
    gpu::GpuMemOwner<VarPixel*> outImageGPUVar;
    outImageGPUVar.TransferVec(outImageGPUPtrVar);
    gpu::GpuMemOwner<MskPixel*> outImageGPUMsk;
    outImageGPUMsk.TransferVec(outImageGPUPtrMsk);

    int blockN = gpu::CalcBlockCount( gpu::GetCudaCurSMCount());

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

    gpu::CopyFromGpu(outImageImg.img, outImageGPUPtrImg[0], outImageImg.Size() );

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
        throw LSST_EXCEPT(GpuRuntimeErrorException, "GPU variance calculation failed to run");
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
        throw LSST_EXCEPT(GpuRuntimeErrorException, "GPU mask calculation failed to run");
    }
    gpu::CopyFromGpu(outImageVar.img, outImageGPUPtrVar[0], outImageVar.Size() );
    gpu::CopyFromGpu(outImageMsk.img, outImageGPUPtrMsk[0], outImageMsk.Size() );
}

#define INSTANTIATE_GPU_ConvolutionMI_SpatiallyInvariantKernel(OutPixelT,InPixelT)  \
        template void GPU_ConvolutionMI_SpatiallyInvariantKernel<OutPixelT,InPixelT>( \
                    ImageBuffer<InPixelT>&    inImageImg,  \
                    ImageBuffer<VarPixel>&    inImageVar,  \
                    ImageBuffer<MskPixel>&    inImageMsk,  \
                    ImageBuffer<OutPixelT>&   outImageImg, \
                    ImageBuffer<VarPixel>&    outImageVar, \
                    ImageBuffer<MskPixel>&    outImageMsk, \
                    ImageBuffer<KerPixel>&    kernel   \
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



