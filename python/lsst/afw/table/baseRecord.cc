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

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/afw/table/Flag.h"
#include "lsst/afw/table/Field.h"
#include "lsst/afw/table/SchemaMapper.h"
#include "lsst/afw/table/BaseColumnView.h"
#include "lsst/afw/table/BaseRecord.h"

namespace py = pybind11;

namespace lsst {
namespace afw {
namespace table {

template <typename T>
void declareBaseRecordOverloads(py::class_<BaseRecord, std::shared_ptr<BaseRecord>> & clsBaseRecord,
                                std::string const & suffix) {
    clsBaseRecord.def(("_get_"+suffix).c_str(),
                      (typename Field<T>::Value (BaseRecord::*)(Key<T> const &) const) &BaseRecord::get);
    clsBaseRecord.def("_getitem_", [](BaseRecord & self, Key<T> const & key)->typename Field<T>::Reference {
        /*
        Define the python __getitem__ method in python to return a baserecord for the requested key
        */
        return self[key];
    });
}

template <typename U>
void declareBaseRecordArrayOverloads(py::class_<BaseRecord, std::shared_ptr<BaseRecord>> clsBaseRecord,
                                     std::string const & suffix) {
    typedef lsst::afw::table::Array<U> T;
    clsBaseRecord.def(("_get_"+suffix).c_str(),
                      (typename Field<T>::Value (BaseRecord::*)(Key<T> const &) const) &BaseRecord::get);
    clsBaseRecord.def("_getitem_", [](BaseRecord & self, Key<T> const & key)->ndarray::Array<U,1,1> {
        /*
        Define the python __getitem__ method in python to return a baserecord for the requested key
        */
        return self[key];
    });
}

template <typename T>
void declareBaseRecordOverloadsFlag(py::class_<BaseRecord, std::shared_ptr<BaseRecord>> clsBaseRecord,
                                    std::string const & suffix) {
    clsBaseRecord.def(("_get_"+suffix).c_str(),
                      (typename Field<T>::Value (BaseRecord::*)(Key<T> const &) const) &BaseRecord::get);
}

template <typename T, typename U>
void declareBaseRecordSet(py::class_<BaseRecord, std::shared_ptr<BaseRecord>> clsBaseRecord,
                          std::string const & suffix) {
    clsBaseRecord.def(("set"+suffix).c_str(), (void (BaseRecord::*)(Key<T> const &, U const &))
        &BaseRecord::set);
};

template <typename U>
void declareBaseRecordSetArray(py::class_<BaseRecord, std::shared_ptr<BaseRecord>> clsBaseRecord,
                               std::string const & suffix) {
    clsBaseRecord.def(("set"+suffix).c_str(),
                      (void (BaseRecord::*)(Key<lsst::afw::table::Array<U>> const &,
                                            ndarray::Array<U,1,1> const &)) &BaseRecord::set);
};

PYBIND11_PLUGIN(_baseRecord) {
    py::module mod("_baseRecord", "Python wrapper for afw _baseRecord library");
    
    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
    };

    /* Module level */
    py::class_<BaseRecord, std::shared_ptr<BaseRecord>> clsBaseRecord(mod, "BaseRecord");

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */
    clsBaseRecord.def("assign", (void (BaseRecord::*)(BaseRecord const &)) &BaseRecord::assign);
    clsBaseRecord.def("assign",(void (BaseRecord::*)(BaseRecord const &, SchemaMapper const &))
        &BaseRecord::assign);
    clsBaseRecord.def("getSchema", &BaseRecord::getSchema);
    clsBaseRecord.def("getTable", &BaseRecord::getTable);
    
    declareBaseRecordOverloads<std::uint16_t>(clsBaseRecord, "U");
    declareBaseRecordOverloads<std::int32_t>(clsBaseRecord, "I");
    declareBaseRecordOverloads<std::int64_t>(clsBaseRecord, "L");
    declareBaseRecordOverloads<float>(clsBaseRecord, "F");
    declareBaseRecordOverloads<double>(clsBaseRecord, "D");
    declareBaseRecordOverloads<std::string>(clsBaseRecord, "String");
    declareBaseRecordOverloadsFlag<lsst::afw::table::Flag>(clsBaseRecord, "Flag");
    declareBaseRecordOverloads<lsst::afw::geom::Angle>(clsBaseRecord, "Angle");
    declareBaseRecordArrayOverloads<std::uint16_t>(clsBaseRecord, "ArrayU");
    declareBaseRecordArrayOverloads<int>(clsBaseRecord, "ArrayI");
    declareBaseRecordArrayOverloads<float>(clsBaseRecord, "ArrayF");
    declareBaseRecordArrayOverloads<double>(clsBaseRecord, "ArrayD");

    declareBaseRecordSet<std::uint16_t, std::uint16_t>(clsBaseRecord, "U");
    declareBaseRecordSet<std::int32_t, std::int32_t>(clsBaseRecord, "I");
    declareBaseRecordSet<std::int64_t, std::int64_t>(clsBaseRecord, "L");
    declareBaseRecordSet<float, float>(clsBaseRecord, "F");
    declareBaseRecordSet<double, double>(clsBaseRecord, "D");
    declareBaseRecordSet<std::string, std::string>(clsBaseRecord, "String");
    declareBaseRecordSet<lsst::afw::table::Flag, bool>(clsBaseRecord, "Flag");
    declareBaseRecordSet<lsst::afw::geom::Angle, lsst::afw::geom::Angle>(clsBaseRecord, "Angle");
    declareBaseRecordSetArray<std::uint16_t>(clsBaseRecord, "ArrayU");
    declareBaseRecordSetArray<int>(clsBaseRecord, "ArrayI");
    declareBaseRecordSetArray<float>(clsBaseRecord, "ArrayF");
    declareBaseRecordSetArray<double>(clsBaseRecord, "ArrayD");

    return mod.ptr();
}

}}}  // namespace lsst::afw::table
