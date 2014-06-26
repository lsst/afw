// -*- LSST-C++ -*-

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
 * @file
 *
 * @brief Functions to query the properties of currently selected GPU device
 *
 * Functions in this file are used to query GPU device.
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

#include "lsst/pex/exceptions.h"
#include "lsst/afw/gpu/GpuExceptions.h"
#include "lsst/afw/gpu/detail/CudaQueryDevice.h"

using namespace lsst::afw::gpu;

#ifndef GPU_BUILD //if no GPU support, throw exceptions

#include <stdio.h>

namespace lsst {
namespace afw {
namespace gpu {
namespace detail {

void PrintCudaDeviceInfo() {
    printf("Afw not compiled with GPU support\n");
}

int GetCudaCurDeviceId() {
    throw LSST_EXCEPT(GpuRuntimeError, "AFW not built with GPU support");
}

int GetCudaCurSMSharedMemorySize() {
    throw LSST_EXCEPT(GpuRuntimeError, "AFW not built with GPU support");
}

int GetCudaCurGlobalMemorySize() {
    throw LSST_EXCEPT(GpuRuntimeError, "AFW not built with GPU support");
}

int GetCudaCurSMRegisterCount() {
    throw LSST_EXCEPT(GpuRuntimeError, "AFW not built with GPU support");
}

int GetCudaCurSMCount() {
    throw LSST_EXCEPT(GpuRuntimeError, "AFW not built with GPU support");
}

bool GetCudaCurIsDoublePrecisionSupported() {
    throw LSST_EXCEPT(GpuRuntimeError, "AFW not built with GPU support");
}

}
}
}
}

#else

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace std;

namespace lsst {
namespace afw {
namespace gpu {
namespace detail {

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


}
}
}
}

#endif

