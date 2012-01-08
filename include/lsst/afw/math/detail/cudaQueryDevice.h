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
 * @brief Exposes some simple procedures to query and select GPU devices
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

namespace lsst {
namespace afw {
namespace math {
namespace detail {
namespace gpu {

/// prints some cuda device information to stdout
void PrintCudaDeviceInfo();

/// returns device ID of currently selected cuda device
int GetCudaCurDeviceId();

/// returns shared memory size per block of currently selected cuda device
int GetCudaCurSMSharedMemorySize();

/// returns global memory size of currently selected cuda device
int GetCudaCurGlobalMemorySize();

/// returns number of registers per block of currently selected cuda device
int GetCudaCurSMRegisterCount();

/// returns number streaming multiprocessors of currently selected cuda device
int GetCudaCurSMCount();

/// returns whether currently selected cuda device supports double precision
bool GetCudaCurIsDoublePrecisionSupported();

/// selects a cuda device
void SetCudaDevice(int devId);

/// reserves cuda device
void CudaReserveDevice();

/// frees resources and releases current cuda device
void CudaThreadExit();

} //namespace gpu ends

}
}
}
} //namespace lsst::afw::math::detail ends

