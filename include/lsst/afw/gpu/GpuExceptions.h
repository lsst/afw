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
