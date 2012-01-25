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


#ifndef GPU_BUILD //if no GPU support, throw exceptions

#include <stdio.h>
#include "lsst/afw/math/detail/Convolve.h"

#include "lsst/afw/math/detail/CudaHelpers.h"

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

#include "lsst/afw/math/detail/Convolve.h"
#include "lsst/afw/math/detail/ImageBuffer.h"

#include "lsst/afw/math/detail/CudaHelpers.h"

using namespace std;

namespace lsst {
namespace afw {
namespace math {
namespace detail {
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

}}}}} // namespace lsst::afw::math::detail::gpu ends

#endif

