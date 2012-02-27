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
 * @brief Functions to help managing setup for GPU kernels
 *
 * Functions in this file are used to query GPU device,
 * and to simplify GPu device selection 
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

/* requires:
#include <cuda.h>
#include <cuda_runtime.h>
#include "lsst/afw/math/detail/ImageBuffer.h"
*/

namespace lsst {
namespace afw {
namespace gpu {
namespace detail {


/// selects a cuda device
void SetCudaDevice(int devId);

/// reserves cuda device
void CudaReserveDevice();

/// frees resources and releases current cuda device
void CudaThreadExit();

// returns true when preffered device has been selected
// returns false when there is no preffered device
// throws exception when unable to select preffered device
bool SelectPreferredCudaDevice();

// throws exception when automatic selection fails
void AutoSelectCudaDevice();

// verifies basic parameters of Cuda device
void VerifyCudaDevice();

bool TryToSelectCudaDevice(const lsst::afw::gpu::DevicePreference devPref);
int GetPreferredCudaDevice();

}}}} //namespace lsst::afw::math::detail::gpu ends

