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

#include "lsst/afw/geom/ellipses/Quadrupole.h"

#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/geom/Angle.h"
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

template <typename T>
void declarePointKey(py::module &mod, std::string const & suffix) {
    py::class_<PointKey<T>> clsPointKey(mod, ("Point"+suffix+"Key").c_str());
    /* Constructors */
    clsPointKey.def(py::init<>());
    clsPointKey.def(py::init<Key<T> const &, Key<T> const &>());
    clsPointKey.def(py::init<SubSchema const &>());
    /* Operators */
    clsPointKey.def(py::self == py::self);
    clsPointKey.def(py::self != py::self);
    /* Members */
    clsPointKey.def("getX", &PointKey<T>::getX);
    clsPointKey.def("getY", &PointKey<T>::getY);
    clsPointKey.def("isValid", &PointKey<T>::isValid);
    clsPointKey.def_static("addFields", &PointKey<T>::addFields);
    clsPointKey.def("set",
                    [](PointKey<T> & self, BaseRecord & record, lsst::afw::geom::Point<T,2> const & value) {
        return self.set(record, value);
    });
    clsPointKey.def("get", &PointKey<T>::get);
};

template <typename T, int N>
void declareCovarianceMatrixKey(py::module &mod, const::std::string & suffix) {
    typedef std::vector< Key<T> > SigmaKeyArray;
    typedef std::vector< Key<T> > CovarianceKeyArray;
    typedef std::vector<std::string> NameArray;
    
    py::class_<CovarianceMatrixKey<T,N>> cls(mod, ("CovarianceMatrix"+suffix+"Key").c_str());
    
    cls.def(py::init<>());
    cls.def(py::init<SigmaKeyArray const &, CovarianceKeyArray const &>(),
            "sigma"_a, "cov"_a=CovarianceKeyArray());
    cls.def(py::init<SubSchema const &, NameArray const &>());
    
    cls.def_static("addFields", (CovarianceMatrixKey<T,N> (*)(
            Schema &,
            std::string const &,
            NameArray const &,
            std::string const &,
            bool)) &CovarianceMatrixKey<T,N>::addFields,
        "schema"_a, "prefix"_a, "names"_a, "unit"_a, "diagonalOnly"_a=false
    );
    cls.def_static("addFields", (CovarianceMatrixKey<T,N> (*)(
            Schema &,
            std::string const &,
            NameArray const &,
            NameArray const &,
            bool)) &CovarianceMatrixKey<T,N>::addFields,
        "schema"_a, "prefix"_a, "names"_a, "units"_a, "diagonalOnly"_a=false
    );
    cls.def("set",
            [](CovarianceMatrixKey<T,N> & cov, BaseRecord & record, Eigen::Matrix<T,N,N> const & value) {
        return cov.set(record, value);
    });
    cls.def("get", [](CovarianceMatrixKey<T,N> & cov, BaseRecord const & record) {
        return cov.get(record);
    });
    cls.def("isValid", &CovarianceMatrixKey<T,N>::isValid);
    cls.def("setElement", &CovarianceMatrixKey<T,N>::setElement);
    cls.def("getElement", &CovarianceMatrixKey<T,N>::getElement);
    cls.def("__eq__", [](CovarianceMatrixKey<T,N> & self, CovarianceMatrixKey<T,N> & other) {
        return self==other;
    });
    cls.def("__ne__", [](CovarianceMatrixKey<T,N> & self, CovarianceMatrixKey<T,N> & other) {
        return self!=other;
    });
};

PYBIND11_PLUGIN(_aggregates) {
    py::module mod("_aggregates", "Python wrapper for afw _aggregates library");

    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
    };

    /* Module level */
    declarePointKey<double>(mod, "2D");
    declarePointKey<int>(mod, "2I");
    declareCovarianceMatrixKey<float,2>(mod, "2f");
    declareCovarianceMatrixKey<float,3>(mod, "3f");
    declareCovarianceMatrixKey<float,4>(mod, "4f");
    declareCovarianceMatrixKey<float,Eigen::Dynamic>(mod, "Xf");
    declareCovarianceMatrixKey<double,2>(mod, "2d");
    declareCovarianceMatrixKey<double,3>(mod, "3d");
    declareCovarianceMatrixKey<double,4>(mod, "4d");
    declareCovarianceMatrixKey<double,Eigen::Dynamic>(mod, "Xd");

    py::class_<CoordKey, FunctorKey<lsst::afw::coord::IcrsCoord>> clsCoordKey(mod, "CoordKey");
    
    py::class_<QuadrupoleKey> clsQuadrupoleKey(mod, "QuadrupoleKey");
    
    py::class_<EllipseKey, std::shared_ptr<EllipseKey>> clsEllipseKey(mod, "EllipseKey");

    /* Member types and enums */

    /* Constructors */
    clsCoordKey.def(py::init<>());
    clsCoordKey.def(py::init<Key<lsst::afw::geom::Angle>, Key<lsst::afw::geom::Angle>>());
    clsCoordKey.def(py::init<SubSchema const &>());
    
    clsQuadrupoleKey.def(py::init<>());
    clsQuadrupoleKey.def(py::init<Key<double> const &, Key<double> const &, Key<double> const &>());
    clsQuadrupoleKey.def(py::init<SubSchema const &>());
    
    clsEllipseKey.def(py::init<>());
    clsEllipseKey.def(py::init<QuadrupoleKey const &, PointKey<double> const &>());
    clsEllipseKey.def(py::init<SubSchema const &>());

    /* Operators */

    /* Members */
    clsCoordKey.def_static("addFields", &CoordKey::addFields);
    clsCoordKey.def("getRa", &CoordKey::getRa);
    clsCoordKey.def("getDec", &CoordKey::getDec);
    clsCoordKey.def("__eq__", [](CoordKey & self, CoordKey & other) {
        return self==other;
    });
    clsCoordKey.def("__ne__", [](CoordKey & self, CoordKey & other) {
        return self!=other;
    });
    clsCoordKey.def("isValid", &CoordKey::isValid);
    clsCoordKey.def("get", [](CoordKey & self, BaseRecord const & record) {
        return self.get(record);
    });
    clsCoordKey.def("set",
                    (void (CoordKey::*)(BaseRecord & record, 
                                        lsst::afw::coord::Coord const & value) const) &CoordKey::set);
    clsCoordKey.def("set",
                    (void (CoordKey::*)(BaseRecord & record,
                                        lsst::afw::coord::IcrsCoord const & value) const) &CoordKey::set);

    clsQuadrupoleKey.def_static("addFields", &QuadrupoleKey::addFields);
    clsQuadrupoleKey.def("getIxx", &QuadrupoleKey::getIxx);
    clsQuadrupoleKey.def("getIyy", &QuadrupoleKey::getIyy);
    clsQuadrupoleKey.def("getIxy", &QuadrupoleKey::getIxy);
    clsQuadrupoleKey.def("isValid", &QuadrupoleKey::isValid);
    clsQuadrupoleKey.def("set", &QuadrupoleKey::set);
    clsQuadrupoleKey.def("get", &QuadrupoleKey::get);
    clsQuadrupoleKey.def("__eq__", [](QuadrupoleKey & self, QuadrupoleKey & other) {
        return self==other;
    });
    clsQuadrupoleKey.def("__ne__", [](QuadrupoleKey & self, QuadrupoleKey & other) {
        return self!=other;
    });
    
    clsEllipseKey.def_static("addFields", &EllipseKey::addFields);
    clsEllipseKey.def("get", &EllipseKey::get);
    clsEllipseKey.def("set", &EllipseKey::set);
    clsEllipseKey.def("isValid", &EllipseKey::isValid);
    clsEllipseKey.def("getCore", &EllipseKey::getCore);
    clsEllipseKey.def("getCenter", &EllipseKey::getCenter);
    clsEllipseKey.def("__eq__", [](EllipseKey & self, EllipseKey & other) {
        return self==other;
    });
    clsEllipseKey.def("__ne__", [](EllipseKey & self, EllipseKey & other) {
        return self!=other;
    });
    
    return mod.ptr();
}

}}}  // namespace lsst::afw::table
