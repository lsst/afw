// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#ifndef LSST_AFW_MATH_DETAIL_GPU_EXCEPTIONS_H
#define LSST_AFW_MATH_DETAIL_GPU_EXCEPTIONS_H
/**
 * @file
 *
 * @brief additional GPU exceptions
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

#include "lsst/pex/exceptions.h"

namespace lsst {
namespace afw {
namespace gpu {

LSST_EXCEPTION_TYPE(GpuMemoryError, lsst::pex::exceptions::RuntimeError, lsst::afw::gpu::GpuMemoryError)
LSST_EXCEPTION_TYPE(GpuRuntimeError, lsst::pex::exceptions::RuntimeError, lsst::afw::gpu::GpuRuntimeError)

}
}
}

#endif
