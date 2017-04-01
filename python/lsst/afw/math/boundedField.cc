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
#include <pybind11/stl.h>

#include <memory>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/math/BoundedField.h"

namespace py = pybind11;

using namespace lsst::afw::math;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace math {
namespace {

using PyClass = py::class_<BoundedField, std::shared_ptr<BoundedField>, lsst::afw::table::io::Persistable>;

template <typename PixelT>
void declareTemplates(PyClass &cls) {
    cls.def("fillImage", &BoundedField::fillImage<PixelT>, "image"_a, "overlapOnly"_a = false);
    cls.def("addToImage", &BoundedField::addToImage<PixelT>, "image"_a, "scaleBy"_a = 1.0,
            "overlapOnly"_a = false);
    cls.def("multiplyImage", &BoundedField::multiplyImage<PixelT>, "image"_a, "overlapOnly"_a = false);
    cls.def("divideImage", &BoundedField::divideImage<PixelT>, "image"_a, "overlapOnly"_a = false);
}

PYBIND11_PLUGIN(_boundedField) {
    py::module mod("_boundedField", "Python wrapper for afw _boundedField library");

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    };

    PyClass cls(mod, "BoundedField");

    cls.def("__mul__", &BoundedField::operator*);
    cls.def("__truediv__", &BoundedField::operator/);

    cls.def("evaluate", (double (BoundedField::*)(double, double) const) & BoundedField::evaluate);
    cls.def("evaluate",
            (ndarray::Array<double, 1, 1> (BoundedField::*)(ndarray::Array<double const, 1> const &,
                                                            ndarray::Array<double const, 1> const &) const) &
                    BoundedField::evaluate);
    cls.def("evaluate",
            (double (BoundedField::*)(lsst::afw::geom::Point2D const &) const) & BoundedField::evaluate);
    cls.def("integrate", &BoundedField::integrate);
    cls.def("mean", &BoundedField::mean);
    cls.def("getBBox", &BoundedField::getBBox);

    // Pybind11 resolves overloads by picking the first one that might work
    declareTemplates<double>(cls);
    declareTemplates<float>(cls);

    cls.def("__str__", [](BoundedField const &self) {
        std::ostringstream os;
        os << self;
        return os.str();
    });
    cls.def("__repr__", [](BoundedField const &self) {
        std::ostringstream os;
        os << "BoundedField(" << self << ")";
        return os.str();
    });

    return mod.ptr();
}

}  // <anonymous>
}  // math
}  // afw
}  // lsst
