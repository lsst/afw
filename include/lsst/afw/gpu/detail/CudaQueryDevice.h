// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file
 *
 * @brief Functions to query the properties of currently selected GPU device
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

namespace lsst {
namespace afw {
namespace gpu {
namespace detail {

/// prints some cuda device information to stdout
void PrintCudaDeviceInfo();

/// returns device ID of currently selected cuda device
int GetCudaCurDeviceId();

/// returns shared memory size per block of currently selected cuda device
int GetCudaCurSMSharedMemorySize();

/// returns global memory size of currently selected cuda device
int GetCudaCurGlobalMemorySize();

/// returns the number of registers per block of currently selected cuda device
int GetCudaCurSMRegisterCount();

/// returns the number of streaming multiprocessors of currently selected cuda device
int GetCudaCurSMCount();

/// returns whether currently selected cuda device supports double precision
bool GetCudaCurIsDoublePrecisionSupported();

}
}
}
} //namespace lsst::afw::gpu::detail ends

