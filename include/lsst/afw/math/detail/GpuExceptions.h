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
 
#ifndef LSST_AFW_MATH_DETAIL_GPU_EXCEPTIONS_H
#define LSST_AFW_MATH_DETAIL_GPU_EXCEPTIONS_H
/**
 * @file
 *
 * @brief additional Gpu exceptions
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */

namespace lsst {
namespace afw {
namespace math {

LSST_EXCEPTION_TYPE(GpuMemoryException, lsst::pex::exceptions::RuntimeErrorException, lsst::afw::math::GpuMemoryException)
LSST_EXCEPTION_TYPE(GpuRuntimeErrorException, lsst::pex::exceptions::RuntimeErrorException, lsst::afw::math::GpuRuntimeErrorException)

}
}
}

#endif
