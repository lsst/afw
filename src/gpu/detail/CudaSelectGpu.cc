// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file
 *
 * @brief Functions for simplifying selecting GPU device, implementation file
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

#include "lsst/pex/exceptions.h"
#include "lsst/afw/gpu/GpuExceptions.h"
#include "lsst/afw/gpu/detail/CudaSelectGpu.h"
#include "lsst/afw/gpu/detail/CudaQueryDevice.h"

using namespace lsst::afw::gpu;

#ifndef GPU_BUILD //if no GPU support, throw exceptions

namespace lsst {
namespace afw {
namespace gpu {
namespace detail {


void SetCudaDevice(int devId) {
    throw LSST_EXCEPT(GpuRuntimeError, "AFW not built with GPU support");
}

void CudaReserveDevice() {
    throw LSST_EXCEPT(GpuRuntimeError, "AFW not built with GPU support");
}

void CudaThreadExit() {
    throw LSST_EXCEPT(GpuRuntimeError, "AFW not built with GPU support");
}

bool SelectPreferredCudaDevice()
{
    throw LSST_EXCEPT(GpuRuntimeError, "AFW not built with gpu support");
}
void AutoSelectCudaDevice()
{
    throw LSST_EXCEPT(GpuRuntimeError, "AFW not built with gpu support");
}
void VerifyCudaDevice()
{
    throw LSST_EXCEPT(GpuRuntimeError, "AFW not built with gpu support");
}
bool TryToSelectCudaDevice(bool noExceptions, bool reselect)
{
    throw LSST_EXCEPT(GpuRuntimeError, "AFW not built with gpu support");
}
int GetPrefferedCudaDevice()
{
    throw LSST_EXCEPT(GpuRuntimeError, "AFW not built with gpu support");
}

}
}
}
}

#else

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <memory.h>



namespace lsst {
namespace afw {
namespace gpu {
namespace detail {

//from CudaQueryDevice.cc
void PrintCudaErrorInfo(cudaError_t cudaError, const char* errorStr);

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
            throw LSST_EXCEPT(GpuRuntimeError, errorStr);
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
        throw LSST_EXCEPT(GpuRuntimeError, "No CUDA capable GPUs found");
    }

    cudaDeviceProp prop = GetDesiredDeviceProperties();
    char errorStr[1000];

    int devId;
    cudaError_t cudaError = cudaChooseDevice(&devId, &prop);
    //printf("Error device %d:\n %s\n", devId, cudaGetErrorString(err));
    if (cudaError != cudaSuccess) {
        throw LSST_EXCEPT(GpuRuntimeError, "Error choosing device automatically");
    }
    cudaError = cudaSetDevice(devId);
    if (cudaError == cudaErrorSetOnActiveProcess) {
        cudaGetLastError(); //clear error
        cudaGetDevice(&devId);
    } else if (cudaError != cudaSuccess) {
        cudaGetLastError(); //clear error
        sprintf(errorStr, "Error automatically selecting device %d:\n %s\n",
                devId, cudaGetErrorString(cudaError));
        throw LSST_EXCEPT(GpuRuntimeError, errorStr);
    }
}

void VerifyCudaDevice()
{
    cudaDeviceProp prop = GetDesiredDeviceProperties();
    char errorStr[1000];

    int devId;
    cudaError_t cudaError = cudaGetDevice(&devId);
    if (cudaError != cudaSuccess) {
        throw LSST_EXCEPT(GpuRuntimeError, "Could not get selected CUDA device ID");
    }
    cudaDeviceProp deviceProp;
    cudaError = cudaGetDeviceProperties(&deviceProp, devId);
    if (cudaError != cudaSuccess) {
        throw LSST_EXCEPT(GpuRuntimeError, "Could not get CUDA device properties");
    }
    if (deviceProp.major < prop.major ||
            (deviceProp.major == prop.major && deviceProp.minor < prop.minor)
       ) {
        sprintf(errorStr, "Only SM %d.%d or better GPU devices are currently allowed", prop.major, prop.minor);
        throw LSST_EXCEPT(GpuRuntimeError, errorStr );
    }

    if (deviceProp.major == prop.major && deviceProp.minor < prop.minor) {
        if (deviceProp.totalGlobalMem < prop.totalGlobalMem) {
            throw LSST_EXCEPT(GpuRuntimeError, "Not enough global memory on GPU");
        }
    }
    if (deviceProp.sharedMemPerBlock < 16 * 1000) {
        throw LSST_EXCEPT(GpuRuntimeError, "Not enough shared memory on GPU");
    }
    if (deviceProp.regsPerBlock < prop.regsPerBlock) {
        throw LSST_EXCEPT(GpuRuntimeError, "Not enough registers per block available on GPU");
    }
    if (deviceProp.maxThreadsPerBlock < prop.maxThreadsPerBlock) {
        throw LSST_EXCEPT(GpuRuntimeError, "Not enough threads per block available on GPU");
    }
}

bool TryToSelectCudaDevice(bool noExceptions, bool reselect)
{
#if !defined(GPU_BUILD)
    return false;
#else
    static bool isDeviceSelected = false;
    static bool isDeviceOk = false;

    if (reselect){
        isDeviceSelected = false;
        isDeviceOk = false;
    }

    if (isDeviceSelected)
        return isDeviceOk;
    isDeviceSelected = true;


    if (!noExceptions) {
        bool done = SelectPreferredCudaDevice();
        if (done) {
            isDeviceOk = true;
            return true;
        }
    } else {
        try {
            bool done = SelectPreferredCudaDevice();
            if (done) {
                isDeviceOk = true;
                return true;
            }
        } catch(...) {
            return false;
        }
    }

    if (!noExceptions) {
        AutoSelectCudaDevice();
        VerifyCudaDevice();
        isDeviceOk = true;
        return true;
    }

    try {
        AutoSelectCudaDevice();
        VerifyCudaDevice();
    } catch(...) {
        return false;
    }

    isDeviceOk = true;
    return true;
#endif
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

}
}
}
} // namespace lsst::afw::gpu::detail ends

#endif


