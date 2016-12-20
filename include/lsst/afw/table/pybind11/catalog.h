#ifndef AFW_TABLE_PYBIND11_CATALOG_H_INCLUDED
#define AFW_TABLE_PYBIND11_CATALOG_H_INCLUDED
/*
 * LSST Data Management System
 * Copyright 2016  AURA/LSST.
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
#include <cstddef>
#include <memory>
#include <string>
#include <sstream>
#include <stdexcept>

#include <pybind11/pybind11.h>
//#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/BaseColumnView.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/Catalog.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace table {

/**
Declare field-type-specific overloaded catalog member functions for one field type

@tparam RecordT  Record type, e.g. BaseRecord or SimpleRecord.
@tparam T  Field type.

@param[in] cls  Catalog pybind11 class.
@param[in] suffix  Field type suffix, such as "I" for int, "L" for long,
    "F" for float, "D" for double, or "Angle" for afw::geom::Angle.
*/
template <typename RecordT, typename T>
void declareCatalogOverloads(
    py::class_<CatalogT<RecordT>, std::shared_ptr<CatalogT<RecordT>>> & cls,
    const std::string suffix
) {
    typedef CatalogT<RecordT> Catalog;

    cls.def("isSorted", (bool (Catalog::*)(Key<T> const &) const) &Catalog::isSorted);
    cls.def("sort", (void (Catalog::*)(Key<T> const &)) &Catalog::sort);
    cls.def(("_find_" + suffix).c_str(),
            [](Catalog & self, T const & value, Key<T> const & key)->PTR(RecordT) {
        typename Catalog::const_iterator iter = self.find(value, key);
        if (iter == self.end()) {
            return PTR(RecordT)();
        };
        return iter;
    });
    cls.def(("_upper_bound_" + suffix).c_str(),
            [](Catalog & self, T const & value, Key<T> const & key)->int {
        return self.upper_bound(value, key) - self.begin();
    });
    cls.def(("_lower_bound_" + suffix).c_str(),
            [](Catalog & self, T const & value, Key<T> const & key)->int {
        return self.lower_bound(value, key) - self.begin();
    });
    cls.def(("_equal_range_" + suffix).c_str(),
            [](Catalog & self, T const & value, Key<T> const & key)->std::pair<int,int> {
        std::pair<typename Catalog::const_iterator, typename Catalog::const_iterator> p
            = self.equal_range(value, key);
        return std::pair<int,int>(p.first - self.begin(), p.second - self.begin());
    });
};

/**
Declare member and static functions for a given instantiation of lsst::afw::table::CatalogT<RecordT>.

@tparam RecordT  Record type, e.g. BaseRecord or SimpleRecord.

@param[in] cls  Catalog pybind11 class.
*/
template <typename RecordT>
void declareCatalog(py::class_<CatalogT<RecordT>, std::shared_ptr<CatalogT<RecordT>>> & cls) {
    typedef CatalogT<RecordT> Catalog;
    typedef typename RecordT::Table Table;
    typedef typename RecordT::ColumnView ColumnView;
    typedef std::vector<PTR(RecordT)> Internal;
    typedef typename Internal::size_type size_type;
    typedef RecordT & reference;

    /* Constructors */
    cls.def(py::init<Schema const &>());
    cls.def(py::init<PTR(Table) const &>());
    cls.def(py::init<Catalog const &>());

    /* Static Methods */
    cls.def_static("readFits",
                   (Catalog (*)(std::string const &, int, int)) &Catalog::readFits,
                   "filename"_a, "hdu"_a=0, "flags"_a=0);
    cls.def_static("readFits",
                   (Catalog (*)(fits::MemFileManager &, int, int)) &Catalog::readFits,
                   "manager"_a, "hdu"_a=0, "flags"_a=0);
    //cls.def_static("readFits", (Catalog (*)()) &Catalog::readFits);

    /* Methods */
    cls.def("getTable", &Catalog::getTable);
    cls.def("getSchema", &Catalog::getSchema);
    cls.def("getColumnView", &Catalog::getColumnView);
    cls.def("capacity", &Catalog::capacity);
    cls.def("addNew", &Catalog::addNew);
    cls.def("__len__", &Catalog::size);
    cls.def("set", &Catalog::set);
    cls.def("_getitem_", [](Catalog & self, int i) {
        // If the index is less than 0, treat as an offset from the end (the Python convention)
        if (i < 0) {
            i = self.size() + i;
        };
        return self.get(i);
    });
    cls.def("isContiguous", &Catalog::isContiguous);
    cls.def("writeFits",
            (void (Catalog::*)(std::string const &, std::string const &, int) const) &Catalog::writeFits,
            "filename"_a, "mode"_a="w", "flags"_a = 0);
    cls.def("writeFits",
            (void (Catalog::*)(fits::MemFileManager &, std::string const &, int) const) &Catalog::writeFits,
            "manager"_a, "mode"_a="w", "flags"_a=0);
    //cls.def("writeFits", (void (Catalog::*)() const) &Catalog::writeFits);
    cls.def("_extend", [](Catalog & self, Catalog const & other, bool deep){
        self.insert(self.end(), other.begin(), other.end(), deep);
    });
    cls.def("_extend", [](Catalog & self, Catalog const & other, SchemaMapper const & mapper) {
        self.insert(mapper, self.end(), other.begin(), other.end());
    });
    cls.def("append", [](Catalog & self, PTR(RecordT) const & rec) {
        self.push_back(rec);
    });
    cls.def("reserve", &Catalog::reserve);
    cls.def("subset", (Catalog (Catalog::*)(ndarray::Array<bool const,1> const &) const) &Catalog::subset);
    cls.def("subset",
            (Catalog (Catalog::*)(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t) const) &Catalog::subset);
    cls.def("__delitem__", [](Catalog & self, std::ptrdiff_t i) {
        auto const i_orig = i;
        if (i < 0) {
            // index backwards from the end
            i += self.size();
        }
        if (i < 0 || static_cast<std::size_t>(i) >= self.size()) {
            std::ostringstream os;
            os << "Catalog index " << i_orig << " out of range.";
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, os.str());
        }
        self.erase(self.begin() + i);

    });

    declareCatalogOverloads<RecordT, std::int32_t>(cls, "I");
    declareCatalogOverloads<RecordT, std::int64_t>(cls, "L");
    declareCatalogOverloads<RecordT, float>(cls, "F");
    declareCatalogOverloads<RecordT, double>(cls, "D");
    declareCatalogOverloads<RecordT, lsst::afw::geom::Angle>(cls, "Angle");
};

}}} // lsst::afw::table

#endif