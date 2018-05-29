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

#include "ndarray/pybind11.h"

#include "lsst/afw/geom/ellipses/Quadrupole.h"

#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/SpherePoint.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/FunctorKey.h"
#include "lsst/afw/table/aggregates.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace table {
namespace {

// We don't expose base classes (e.g. FunctorKey) to Python, since they're just used to
// define a CRTP interface in C++ and in Python that's just duck-typing.

template <typename T>
using PyPointKey = py::class_<PointKey<T>, std::shared_ptr<PointKey<T>>>;

template <typename Box>
using PyBoxKey = py::class_<BoxKey<Box>, std::shared_ptr<BoxKey<Box>>>;

using PyCoordKey = py::class_<CoordKey, std::shared_ptr<CoordKey>>;

using PyQuadrupoleKey = py::class_<QuadrupoleKey, std::shared_ptr<QuadrupoleKey>>;

using PyEllipseKey = py::class_<EllipseKey, std::shared_ptr<EllipseKey>>;

template <typename T, int N>
using PyCovarianceMatrixKey =
        py::class_<CovarianceMatrixKey<T, N>, std::shared_ptr<CovarianceMatrixKey<T, N>>>;

template <typename T>
static void declarePointKey(py::module &mod, std::string const &suffix) {
    PyPointKey<T> cls(mod, ("Point" + suffix + "Key").c_str());
    cls.def(py::init<>());
    cls.def(py::init<Key<T> const &, Key<T> const &>(), "x"_a, "y"_a);
    cls.def(py::init<SubSchema const &>());
    cls.def("__eq__", &PointKey<T>::operator==, py::is_operator());
    cls.def("__ne__", &PointKey<T>::operator!=, py::is_operator());
    cls.def("getX", &PointKey<T>::getX);
    cls.def("getY", &PointKey<T>::getY);
    cls.def("isValid", &PointKey<T>::isValid);
    cls.def_static("addFields", &PointKey<T>::addFields, "schema"_a, "name"_a, "doc"_a, "unit"_a);
    cls.def("set", [](PointKey<T> &self, BaseRecord &record, geom::Point<T, 2> const &value) {
        return self.set(record, value);
    });
    cls.def("get", &PointKey<T>::get);
};

template <typename Box>
static void declareBoxKey(py::module &mod, std::string const &suffix) {
    using Element = typename Box::Element;
    PyBoxKey<Box> cls(mod, ("Box" + suffix + "Key").c_str());
    cls.def(py::init<>());
    cls.def(py::init<PointKey<Element> const &, PointKey<Element> const &>(), "min"_a, "max"_a);
    cls.def(py::init<SubSchema const &>());
    cls.def("__eq__", &BoxKey<Box>::operator==, py::is_operator());
    cls.def("__ne__", &BoxKey<Box>::operator!=, py::is_operator());
    cls.def("getMin", &BoxKey<Box>::getMin);
    cls.def("getMax", &BoxKey<Box>::getMax);
    cls.def("isValid", &BoxKey<Box>::isValid);
    cls.def_static("addFields", &BoxKey<Box>::addFields, "schema"_a, "name"_a, "doc"_a, "unit"_a);
    cls.def("set", &BoxKey<Box>::set);
    cls.def("get", &BoxKey<Box>::get);
};

static void declareCoordKey(py::module &mod) {
    PyCoordKey cls(mod, "CoordKey");
    cls.def(py::init<>());
    cls.def(py::init<Key<lsst::afw::geom::Angle>, Key<lsst::afw::geom::Angle>>(), "ra"_a, "dec"_a);
    cls.def(py::init<SubSchema const &>());
    cls.def("__eq__", &CoordKey::operator==, py::is_operator());
    cls.def("__ne__", &CoordKey::operator!=, py::is_operator());
    cls.def_static("addFields", &CoordKey::addFields, "schema"_a, "name"_a, "doc"_a);
    cls.def("getRa", &CoordKey::getRa);
    cls.def("getDec", &CoordKey::getDec);
    cls.def("isValid", &CoordKey::isValid);
    cls.def("get", [](CoordKey &self, BaseRecord const &record) { return self.get(record); });
    cls.def("set", &CoordKey::set);
}

static void declareQuadrupoleKey(py::module &mod) {
    PyQuadrupoleKey cls(mod, "QuadrupoleKey");
    cls.def(py::init<>());
    cls.def(py::init<Key<double> const &, Key<double> const &, Key<double> const &>(), "ixx"_a, "iyy"_a,
            "ixy"_a);
    cls.def(py::init<SubSchema const &>());
    cls.def("__eq__", &QuadrupoleKey::operator==, py::is_operator());
    cls.def("__nq__", &QuadrupoleKey::operator!=, py::is_operator());
    cls.def_static("addFields", &QuadrupoleKey::addFields, "schema"_a, "name"_a, "doc"_a,
                   "coordType"_a = CoordinateType::PIXEL);
    cls.def("getIxx", &QuadrupoleKey::getIxx);
    cls.def("getIyy", &QuadrupoleKey::getIyy);
    cls.def("getIxy", &QuadrupoleKey::getIxy);
    cls.def("isValid", &QuadrupoleKey::isValid);
    cls.def("set", &QuadrupoleKey::set);
    cls.def("get", &QuadrupoleKey::get);
}

static void declareEllipseKey(py::module &mod) {
    PyEllipseKey cls(mod, "EllipseKey");
    cls.def(py::init<>());
    cls.def(py::init<QuadrupoleKey const &, PointKey<double> const &>(), "qKey"_a, "pKey"_a);
    cls.def(py::init<SubSchema const &>());
    cls.def("__eq__", &EllipseKey::operator==, py::is_operator());
    cls.def("__nq__", &EllipseKey::operator!=, py::is_operator());
    cls.def_static("addFields", &EllipseKey::addFields, "schema"_a, "name"_a, "doc"_a, "unit"_a);
    cls.def("get", &EllipseKey::get);
    cls.def("set", &EllipseKey::set);
    cls.def("isValid", &EllipseKey::isValid);
    cls.def("getCore", &EllipseKey::getCore);
    cls.def("getCenter", &EllipseKey::getCenter);
}

template <typename T, int N>
static void declareCovarianceMatrixKey(py::module &mod, const ::std::string &suffix) {
    typedef std::vector<Key<T>> SigmaKeyArray;
    typedef std::vector<Key<T>> CovarianceKeyArray;
    typedef std::vector<std::string> NameArray;

    PyCovarianceMatrixKey<T, N> cls(mod, ("CovarianceMatrix" + suffix + "Key").c_str());

    cls.def(py::init<>());
    // Ordering of the next two ctor declaration matters, as a workaround for DM-8580.
    cls.def(py::init<SubSchema const &, NameArray const &>());
    cls.def(py::init<SigmaKeyArray const &, CovarianceKeyArray const &>(), "sigma"_a,
            "cov"_a = CovarianceKeyArray());
    cls.def("__eq__", &CovarianceMatrixKey<T, N>::operator==, py::is_operator());
    cls.def("__ne__", &CovarianceMatrixKey<T, N>::operator!=, py::is_operator());
    cls.def_static("addFields", (CovarianceMatrixKey<T, N>(*)(Schema &, std::string const &,
                                                              NameArray const &, std::string const &, bool)) &
                                        CovarianceMatrixKey<T, N>::addFields,
                   "schema"_a, "prefix"_a, "names"_a, "unit"_a, "diagonalOnly"_a = false);
    cls.def_static("addFields", (CovarianceMatrixKey<T, N>(*)(Schema &, std::string const &,
                                                              NameArray const &, NameArray const &, bool)) &
                                        CovarianceMatrixKey<T, N>::addFields,
                   "schema"_a, "prefix"_a, "names"_a, "units"_a, "diagonalOnly"_a = false);
    cls.def("set", [](CovarianceMatrixKey<T, N> &cov, BaseRecord &record,
                      Eigen::Matrix<T, N, N> const &value) { return cov.set(record, value); });
    cls.def("get", [](CovarianceMatrixKey<T, N> &cov, BaseRecord const &record) { return cov.get(record); });
    cls.def("isValid", &CovarianceMatrixKey<T, N>::isValid);
    cls.def("setElement", &CovarianceMatrixKey<T, N>::setElement);
    cls.def("getElement", &CovarianceMatrixKey<T, N>::getElement);
};

}  // namespace lsst::afw::table::<anonymous>

PYBIND11_PLUGIN(aggregates) {
    py::module::import("lsst.afw.geom.ellipses");
    py::module::import("lsst.afw.table.base");
    py::module mod("aggregates");

    py::enum_<CoordinateType>(mod, "CoordinateType")
            .value("PIXEL", CoordinateType::PIXEL)
            .value("CELESTIAL", CoordinateType::CELESTIAL)
            .export_values();

    declarePointKey<double>(mod, "2D");
    declarePointKey<int>(mod, "2I");

    declareBoxKey<geom::Box2D>(mod, "2D");
    declareBoxKey<geom::Box2I>(mod, "2I");

    declareCoordKey(mod);
    declareQuadrupoleKey(mod);
    declareEllipseKey(mod);

    declareCovarianceMatrixKey<float, 2>(mod, "2f");
    declareCovarianceMatrixKey<float, 3>(mod, "3f");
    declareCovarianceMatrixKey<float, 4>(mod, "4f");
    declareCovarianceMatrixKey<float, Eigen::Dynamic>(mod, "Xf");
    declareCovarianceMatrixKey<double, 2>(mod, "2d");
    declareCovarianceMatrixKey<double, 3>(mod, "3d");
    declareCovarianceMatrixKey<double, 4>(mod, "4d");
    declareCovarianceMatrixKey<double, Eigen::Dynamic>(mod, "Xd");

    return mod.ptr();
}
}
}
}  // namespace lsst::afw::table
