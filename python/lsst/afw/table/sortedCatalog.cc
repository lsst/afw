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

#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/SortedCatalog.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace lsst::afw::table;

template <typename RecordT>
void declareSortedCatalog(py::module & mod, const std::string & prefix){
    typedef typename RecordT::Table Table;
    py::class_<SortedCatalogT<RecordT>,
               std::shared_ptr<SortedCatalogT<RecordT>>,
               CatalogT<RecordT>> clsSortedCatalog(mod, (prefix+"CatalogBase").c_str());
    clsSortedCatalog.def(py::init<PTR(Table) const &>(),
                         "table"_a=PTR(Table)());
    clsSortedCatalog.def("isSorted", (bool (SortedCatalogT<RecordT>::*)() const)
        &SortedCatalogT<RecordT>::isSorted);
};

PYBIND11_PLUGIN(_sortedCatalog) {
    py::module mod("_sortedCatalog", "Python wrapper for afw _sortedCatalog library");

    /* Module level */
    declareSortedCatalog<SourceRecord>(mod, "Source");
    declareSortedCatalog<SimpleRecord>(mod, "Simple");

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}