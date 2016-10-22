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
#include <pybind11/operators.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "lsst/afw/geom/AffineTransform.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace lsst::afw::geom;

PYBIND11_PLUGIN(_affineTransform) {
    py::module mod("_affineTransform", "Python wrapper for afw _affineTransform library");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    py::class_<AffineTransform> clsAffineTransform(mod, "AffineTransform");

    /* Constructors */
    clsAffineTransform.def(py::init<>());
    clsAffineTransform.def(py::init<Eigen::Matrix3d const &>());
    clsAffineTransform.def(py::init<Eigen::Matrix2d const &>());
    clsAffineTransform.def(py::init<Eigen::Vector2d const &>());
    clsAffineTransform.def(py::init<Eigen::Matrix2d const &, Eigen::Vector2d const &>());
    clsAffineTransform.def(py::init<LinearTransform const &>());
    clsAffineTransform.def(py::init<Extent2D const &>());
    clsAffineTransform.def(py::init<LinearTransform const &, Extent2D const &>());

    /* Operators */
    clsAffineTransform.def(py::self * py::self);
    clsAffineTransform.def("__call__", [](AffineTransform &t, Point2D const &p) -> Point2D {
            return t(p);
    });
    clsAffineTransform.def("__call__", [](AffineTransform &t, Extent2D const &p) -> Extent2D {
            return t(p);
    });
    // These are used by __getitem__ in Python, this was adopted from the Swig approach,
    // but might actually be done better with pybind11.
    clsAffineTransform.def("_getitem_nochecking", [](AffineTransform &t, int row, int col) -> double {
        return (t.getMatrix())(row, col);
    });
    clsAffineTransform.def("_getitem_nochecking", [](AffineTransform &t, int i) -> double {
        return t.operator[](i);
    });

    /* Members */
    clsAffineTransform.def("invert", &AffineTransform::invert);
    clsAffineTransform.def("isIdentity", &AffineTransform::isIdentity);
    clsAffineTransform.def("getTranslation", (Extent2D & (AffineTransform::*)()) &AffineTransform::getTranslation);
    clsAffineTransform.def("getLinear", (LinearTransform & (AffineTransform::*)()) &AffineTransform::getLinear);
    clsAffineTransform.def("getMatrix", &AffineTransform::getMatrix);
    clsAffineTransform.def("getParameterVector", &AffineTransform::getParameterVector);
    clsAffineTransform.def("setParameterVector", &AffineTransform::setParameterVector);
    clsAffineTransform.def_static("makeScaling", (AffineTransform (*)(double)) &AffineTransform::makeScaling);
    clsAffineTransform.def_static("makeScaling", (AffineTransform (*)(double, double)) &AffineTransform::makeScaling);
    clsAffineTransform.def_static("makeRotation", &AffineTransform::makeRotation, "angle"_a);
    clsAffineTransform.def_static("makeTranslation", &AffineTransform::makeTranslation, "translation"_a);

    /* Non-members */
    mod.def("makeAffineTransformFromTriple", makeAffineTransformFromTriple);

    return mod.ptr();
}
