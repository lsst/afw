/* 
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */

#include <pybind11/pybind11.h>
//#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "lsst/pex/exceptions/Exception.h"
#include "lsst/pex/exceptions/python/Exception.h"
#include "lsst/afw/fits.h"

namespace py = pybind11;

using namespace lsst::afw::fits;

PYBIND11_PLUGIN(_fits) {
    py::module mod("_fits", "Python wrapper for afw _fits library");

    /* Module level */
    py::register_exception<FitsError>(mod, "FitsError");
    py::register_exception<FitsTypeError>(mod, "FitsTypeError");
    py::class_<MemFileManager> clsMemFileManager(mod, "MemFileManager");

    /* Member types and enums */

    /* Constructors */
    clsMemFileManager.def(py::init<>());

    /* Operators */

    /* Members */

    return mod.ptr();
}