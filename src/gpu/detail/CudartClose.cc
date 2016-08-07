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
