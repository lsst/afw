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

#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseColumnView.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/Catalog.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace lsst::afw;

using namespace table;

template <typename RecordT, typename T>
void declareCatalogOverloads(py::class_<CatalogT<RecordT>, std::shared_ptr<CatalogT<RecordT>>> & clsCatalogT,
                             const std::string suffix){
    clsCatalogT.def("isSorted", (bool (CatalogT<RecordT>::*)(Key<T> const &) const)
        &CatalogT<RecordT>::isSorted);
    clsCatalogT.def("sort", (void (CatalogT<RecordT>::*)(Key<T> const &)) &CatalogT<RecordT>::sort);
    clsCatalogT.def(("_find_"+suffix).c_str(), [](CatalogT<RecordT> & self,
                                                  T const & value, Key<T> const & key)->PTR(RecordT){
        typename CatalogT<RecordT>::const_iterator iter = self.find(value, key);
        if (iter == self.end()) {
            return PTR(RecordT)();
        };
        return iter;
    });
    clsCatalogT.def(("_upper_bound_"+suffix).c_str(), [](CatalogT<RecordT> & self,
                                                  T const & value, Key<T> const & key)->int{
        return self.upper_bound(value, key) - self.begin();
    });
    clsCatalogT.def(("_lower_bound_"+suffix).c_str(), [](CatalogT<RecordT> & self,
                                                  T const & value, Key<T> const & key)->int{
        return self.lower_bound(value, key) - self.begin();
    });
    clsCatalogT.def(("_equal_range_"+suffix).c_str(), [](CatalogT<RecordT> & self,
                                                  T const & value, Key<T> const & key)->std::pair<int,int>{
        std::pair<typename CatalogT<RecordT>::const_iterator,typename CatalogT<RecordT>::const_iterator> p
            = self.equal_range(value, key);
        return std::pair<int,int>(p.first - self.begin(), p.second - self.begin());
    });
};

template <typename RecordT>
void declareCatalog(py::module & mod, const std::string & prefix){
    typedef typename RecordT::Table Table;
    typedef typename RecordT::ColumnView ColumnView;
    typedef std::vector<PTR(RecordT)> Internal;
    typedef typename Internal::size_type size_type;
    typedef RecordT & reference;
    
    py::class_<CatalogT<RecordT>, std::shared_ptr<CatalogT<RecordT>>>
        clsCatalogT(mod, (prefix+"Catalog").c_str());
    
    /* Constructors */
    clsCatalogT.def(py::init<Schema const &>());
    clsCatalogT.def(py::init<PTR(Table) const &>());
    clsCatalogT.def(py::init<CatalogT<RecordT> const &>());
    
    /* Members */
    clsCatalogT.def("getTable", &CatalogT<RecordT>::getTable);
    clsCatalogT.def("getSchema", &CatalogT<RecordT>::getSchema);
    clsCatalogT.def("getColumnView", &CatalogT<RecordT>::getColumnView);
    //clsCatalogT.def("isContiguous", &CatalogT<RecordT>::isContiguous);
    clsCatalogT.def("capacity", &CatalogT<RecordT>::capacity);
    clsCatalogT.def("addNew", &CatalogT<RecordT>::addNew);
    clsCatalogT.def("__len__", &CatalogT<RecordT>::size);
    clsCatalogT.def("set", &CatalogT<RecordT>::set);
    clsCatalogT.def("_getitem_", [](CatalogT<RecordT> & self, int i) {
        // If the index is less than 0, use the pythonic index
        if(i<0){
            i = self.size()+i;
        };
        return self.get(i);
    });
    clsCatalogT.def("isContiguous", &CatalogT<RecordT>::isContiguous);
    clsCatalogT.def("writeFits",
                    (void (CatalogT<RecordT>::*)(std::string const &, std::string const &, int) const)
                        &CatalogT<RecordT>::writeFits,
                    "filename"_a, "mode"_a="w", "flags"_a = 0);
    clsCatalogT.def("writeFits",
                    (void (CatalogT<RecordT>::*)(fits::MemFileManager &,
                                                 std::string const &,
                                                 int) const) &CatalogT<RecordT>::writeFits,
                    "manager"_a, "mode"_a="w", "flags"_a=0);
    //clsCatalogT.def("writeFits", (void (CatalogT<RecordT>::*)() const) &CatalogT<RecordT>::writeFits);
    clsCatalogT.def_static("readFits",
                           (CatalogT<RecordT> (*)(std::string const &, int, int))
                               &CatalogT<RecordT>::readFits,
                           "filename"_a, "hdu"_a=0, "flags"_a=0);
    clsCatalogT.def_static("readFits",
                           (CatalogT<RecordT> (*)(fits::MemFileManager &, int, int))
                               &CatalogT<RecordT>::readFits,
                            "manager"_a, "hdu"_a=0, "flags"_a=0);
    //clsCatalogT.def_static("readFits", (CatalogT<RecordT> (*)()) &CatalogT<RecordT>::readFits);
    clsCatalogT.def("_extend", [](CatalogT<RecordT> & self,
                                  CatalogT<RecordT> const & other,
                                  bool deep){
        self.insert(self.end(), other.begin(), other.end(), deep);
    });
    clsCatalogT.def("_extend", [](CatalogT<RecordT> & self,
                                  CatalogT<RecordT> const & other,
                                  SchemaMapper const & mapper){
        self.insert(mapper, self.end(), other.begin(), other.end());
    });
    clsCatalogT.def("append", [](CatalogT<RecordT> & self, PTR(RecordT) const & rec){
        self.push_back(rec);
    });
    clsCatalogT.def("reserve", &CatalogT<RecordT>::reserve);
    clsCatalogT.def("subset", (CatalogT<RecordT>
        (CatalogT<RecordT>::*)(ndarray::Array<bool const,1> const &) const) &CatalogT<RecordT>::subset);
    clsCatalogT.def("subset",
                    (CatalogT<RecordT> (CatalogT<RecordT>::*)(std::ptrdiff_t,
                                                              std::ptrdiff_t,
                                                              std::ptrdiff_t) const)
                                                                  &CatalogT<RecordT>::subset);
    declareCatalogOverloads<RecordT, std::int32_t>(clsCatalogT, "I");
    declareCatalogOverloads<RecordT, std::int64_t>(clsCatalogT, "L");
    declareCatalogOverloads<RecordT, float>(clsCatalogT, "F");
    declareCatalogOverloads<RecordT, double>(clsCatalogT, "D");
    declareCatalogOverloads<RecordT, lsst::afw::geom::Angle>(clsCatalogT, "Angle");
};

PYBIND11_PLUGIN(_catalog) {
    py::module mod("_catalog", "Python wrapper for afw _catalog library");

    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
    };

    /* Module level */

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */
    declareCatalog<BaseRecord>(mod, "Base");
    declareCatalog<SourceRecord>(mod, "Source");
    declareCatalog<SimpleRecord>(mod, "Simple");

    return mod.ptr();
}