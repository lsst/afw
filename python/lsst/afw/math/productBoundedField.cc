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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "lsst/afw/math/ProductBoundedField.h"

namespace py = pybind11;

using namespace lsst::afw::math;

using PyClass = py::class_<ProductBoundedField, std::shared_ptr<ProductBoundedField>, BoundedField>;

PYBIND11_MODULE(productBoundedField, mod) {
    PyClass cls(mod, "ProductBoundedField");
    cls.def(py::init<std::vector<std::shared_ptr<BoundedField const>>>());
    // All other operations are wrapped by the BoundedField base class.
}
