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

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "lsst/afw/geom/AffineTransform.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace geom {
namespace {

using PyAffineTransform = py::class_<AffineTransform>;

PYBIND11_PLUGIN(_affineTransform) {
    py::module mod("_affineTransform");

    py::module::import("lsst.afw.geom._LinearTransform");
    py::module::import("lsst.afw.geom._coordinates");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    PyAffineTransform cls(mod, "AffineTransform");

    // Parameters enum is really only used as integer constants.
    cls.attr("XX") = py::cast(int(AffineTransform::Parameters::XX));
    cls.attr("YX") = py::cast(int(AffineTransform::Parameters::YX));
    cls.attr("XY") = py::cast(int(AffineTransform::Parameters::XY));
    cls.attr("YY") = py::cast(int(AffineTransform::Parameters::YY));
    cls.attr("X") = py::cast(int(AffineTransform::Parameters::X));
    cls.attr("Y") = py::cast(int(AffineTransform::Parameters::Y));

    /* Constructors */
    cls.def(py::init<>());
    cls.def(py::init<Eigen::Matrix3d const &>(), "matrix"_a);
    cls.def(py::init<Eigen::Matrix2d const &>(), "linear"_a);
    cls.def(py::init<Eigen::Vector2d const &>(), "translation"_a);
    cls.def(py::init<Eigen::Matrix2d const &, Eigen::Vector2d const &>(), "linear"_a, "translation"_a);
    cls.def(py::init<LinearTransform const &>(), "linear"_a);
    cls.def(py::init<Extent2D const &>(), "translation"_a);
    cls.def(py::init<LinearTransform const &, Extent2D const &>(), "linear"_a, "translation"_a);

    /* Operators and special methods */
    cls.def("__mul__", &AffineTransform::operator*, py::is_operator());
    cls.def("__call__", [](AffineTransform const &t, Point2D const &p) -> Point2D {
            return t(p);
    });
    cls.def("__call__", [](AffineTransform const &t, Extent2D const &p) -> Extent2D {
            return t(p);
    });
    cls.def("__setitem__", [](AffineTransform & self, int i, double value) {
        if (i < 0 || i > 5) {
            PyErr_Format(PyExc_IndexError, "Invalid index for AffineTransform: %d", i);
            throw py::error_already_set();
        }
        self[i] = value;
    });
    cls.def("__getitem__", [](AffineTransform const & self, int row, int col) {
        if (row < 0 || row > 2 || col < 0 || col > 2) {
            PyErr_Format(PyExc_IndexError, "Invalid index for AffineTransform: %d, %d", row, col);
            throw py::error_already_set();
        }
        return (self.getMatrix())(row, col);
    });
    cls.def("__getitem__", [](AffineTransform const & self, int i) {
        if (i < 0 || i > 5) {
            PyErr_Format(PyExc_IndexError, "Invalid index for AffineTransform: %d", i);
            throw py::error_already_set();
        }
        return self[i];
    });
    cls.def("__str__", [](AffineTransform const & self) {
        return py::str(py::cast(self.getMatrix()));
    });
    cls.def("__repr__", [](AffineTransform const & self) {
        return py::str("AffineTransform(\n{}\n)").format(py::cast(self.getMatrix()));
    });
    cls.def("__reduce__", [cls](AffineTransform const & self) {
        return py::make_tuple(cls, py::make_tuple(py::cast(self.getMatrix())));
    });

    /* Members */
    cls.def("invert", &AffineTransform::invert);
    cls.def("isIdentity", &AffineTransform::isIdentity);
    cls.def("getTranslation", (Extent2D & (AffineTransform::*)()) &AffineTransform::getTranslation);
    cls.def("getLinear", (LinearTransform & (AffineTransform::*)()) &AffineTransform::getLinear);
    cls.def("getMatrix", &AffineTransform::getMatrix);
    cls.def("getParameterVector", &AffineTransform::getParameterVector);
    cls.def("setParameterVector", &AffineTransform::setParameterVector);
    cls.def_static("makeScaling", (AffineTransform (*)(double)) &AffineTransform::makeScaling);
    cls.def_static("makeScaling", (AffineTransform (*)(double, double)) &AffineTransform::makeScaling);
    cls.def_static("makeRotation", &AffineTransform::makeRotation, "angle"_a);
    cls.def_static("makeTranslation", &AffineTransform::makeTranslation, "translation"_a);

    /* Non-members */
    mod.def("makeAffineTransformFromTriple", makeAffineTransformFromTriple);

    return mod.ptr();
}

}}}} // namespace lsst::afw::geom::<anonymous>
