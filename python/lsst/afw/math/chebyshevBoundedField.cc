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
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/pex/config/python.h" // defines LSST_DECLARE_CONTROL_FIELD
#include "lsst/afw/table/io/python.h"

#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/math/BoundedField.h"
#include "lsst/afw/math/ChebyshevBoundedField.h"

namespace py = pybind11;

using namespace lsst::afw::math;

using ClsField = py::class_<ChebyshevBoundedField, std::shared_ptr<ChebyshevBoundedField>,
                            BoundedField>;

template <typename PixelT>
void declareTemplates(ClsField & cls) {
    cls.def_static("fit", (std::shared_ptr<ChebyshevBoundedField>(*)(lsst::afw::image::Image<PixelT> const &,
                                                                     ChebyshevBoundedFieldControl const &)) &
                                  ChebyshevBoundedField::fit);
}

PYBIND11_PLUGIN(_chebyshevBoundedField) {
    py::module mod("_chebyshevBoundedField", "Python wrapper for afw _chebyshevBoundedField library");

    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
    };

    /* Module level */
    py::class_<ChebyshevBoundedFieldControl> clsChebyshevBoundedFieldControl(mod,
                                                                             "ChebyshevBoundedFieldControl");
    ClsField clsChebyshevBoundedField(mod, "ChebyshevBoundedField");

    /* Member types and enums */
    using Control = ChebyshevBoundedFieldControl;

    /* Constructors */
    clsChebyshevBoundedFieldControl.def(py::init<>());
    clsChebyshevBoundedField.def(
            py::init<lsst::afw::geom::Box2I const &, ndarray::Array<double const, 2, 2> const &>());

    /* Operators are defined only in the BoundedField base class;
       we let Python inheritance provide them here. */

    /* Members */
    LSST_DECLARE_CONTROL_FIELD(clsChebyshevBoundedFieldControl, ChebyshevBoundedFieldControl, orderX);
    LSST_DECLARE_CONTROL_FIELD(clsChebyshevBoundedFieldControl, ChebyshevBoundedFieldControl, orderY);
    LSST_DECLARE_CONTROL_FIELD(clsChebyshevBoundedFieldControl, ChebyshevBoundedFieldControl, triangular);

    clsChebyshevBoundedFieldControl.def("computeSize", &ChebyshevBoundedFieldControl::computeSize);

    clsChebyshevBoundedField.def("getCoefficients", &ChebyshevBoundedField::getCoefficients);
    clsChebyshevBoundedField.def_static(
            "fit", (std::shared_ptr<ChebyshevBoundedField>(*)(
                           lsst::afw::geom::Box2I const &, ndarray::Array<double const, 1> const &,
                           ndarray::Array<double const, 1> const &, ndarray::Array<double const, 1> const &,
                           Control const &)) &
                           ChebyshevBoundedField::fit);
    clsChebyshevBoundedField.def_static(
            "fit", (std::shared_ptr<ChebyshevBoundedField>(*)(
                           lsst::afw::geom::Box2I const &, ndarray::Array<double const, 1> const &,
                           ndarray::Array<double const, 1> const &, ndarray::Array<double const, 1> const &,
                           ndarray::Array<double const, 1> const &, Control const &)) &
                           ChebyshevBoundedField::fit);

    clsChebyshevBoundedField.def("truncate", &ChebyshevBoundedField::truncate);

    // Pybind11 resolves overloads by picking the first one that might work
    declareTemplates<double>(clsChebyshevBoundedField);
    declareTemplates<float>(clsChebyshevBoundedField);

    return mod.ptr();
}
