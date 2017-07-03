/*
 * LSST Data Management System
 * Copyright 2017 AURA/LSST.
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

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/pex/config/python.h"  // defines LSST_DECLARE_CONTROL_FIELD
#include "lsst/afw/table/io/python.h"

#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/math/BoundedField.h"
#include "lsst/afw/math/TransformBoundedField.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace math {
namespace {

using ClsField = py::class_<TransformBoundedField, std::shared_ptr<TransformBoundedField>, BoundedField,
                            lsst::afw::table::io::PersistableFacade<TransformBoundedField>>;

PYBIND11_PLUGIN(_transformBoundedField) {
    py::module mod("_transformBoundedField");

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    };

    /* Module level */
    lsst::afw::table::io::python::declarePersistableFacade<TransformBoundedField>(mod,
                                                                                  "TransformBoundedField");
    ClsField cls(mod, "TransformBoundedField");

    cls.def(py::init<lsst::afw::geom::Box2I const &, TransformBoundedField::Transform const &>(), "bbox"_a,
            "transform"_a);

    cls.def("__mul__", &TransformBoundedField::operator*, py::is_operator());
    cls.def("__eq__", &TransformBoundedField::operator==, py::is_operator());

    cls.def("getTransform", &TransformBoundedField::getTransform);
    cls.def("evaluate", (double (BoundedField::*)(double, double) const) & BoundedField::evaluate);
    cls.def("evaluate",
            (ndarray::Array<double, 1, 1>(TransformBoundedField::*)(
                    ndarray::Array<double const, 1> const &, ndarray::Array<double const, 1> const &) const) &
                    TransformBoundedField::evaluate);
    cls.def("evaluate", (double (TransformBoundedField::*)(lsst::afw::geom::Point2D const &) const) &
                                TransformBoundedField::evaluate);

    return mod.ptr();
}

}  // namespace
}  // namespace math
}  // namespace afw
}  // namespace lsst