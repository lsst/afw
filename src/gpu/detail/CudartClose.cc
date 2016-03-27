// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file
 *
 * @brief Calls cudaThreadExit from a destructor of a global object.
 *
 * When using library libcudart.so (CUDA 3.2) from Python, it fails to properly close the library 
 * resulting in segmentation fault. This file is intended to solve that problem by
 * automatically deinitializing the library just prior to exiting.
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

#ifdef GPU_BUILD

#include <cuda.h>
#include <cuda_runtime.h>

namespace {


struct LibCudartCleanup
{
    ~LibCudartCleanup()
    {
        cudaThreadExit();
    }
};

LibCudartCleanup globCudaCleanup;

}

#endif
