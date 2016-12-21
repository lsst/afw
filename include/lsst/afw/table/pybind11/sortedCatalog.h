#ifndef AFW_TABLE_PYBIND11_SORTEDCATALOG_H_INCLUDED
#define AFW_TABLE_PYBIND11_SORTEDCATALOG_H_INCLUDED
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

namespace lsst {
namespace afw {
namespace table {
namespace pybind11 {

/**
Declare member and static functions for a given instantiation of lsst::afw::table::CatalogT<RecordT>
that supplement or override methods of lsst::afw::table::CatalogT<RecordT>

To use this:
- Instantiate a hidden CatalogT<RecordT> unsorted base class, using a name with a leading underscore.
- Call `declareCatalog` to add methods to the hidden unsorted base class.
- Instantiate a SortedCatalogT<RecordT> as the class of interest.
- Call `declareSortedCatalog` to add methods to it.
- In a python module:

    from lsst.afw.table.catalog import addCatalogMethods
    from lsst.afw.table.sortedCatalog import addSortedCatalogMethods
    from *yourmodule* import *YourCatalogClass*
    addCatalogMethods(*YourCatalogClass*)
    addSortedCatalogMethods(*YourCatalogClass*)

@tparam RecordT  Record type, e.g. BaseRecord or SimpleRecord.

@param[in] cls  Catalog pybind11 class.

@warning It is crucial to specify all methods that SortedCatalogT overloads here.
Otherwise the wrong version will be called, which can lead to issues such as
returning instances of the hidden base class.
*/
template <typename RecordT>
void declareSortedCatalog(
    py::class_<SortedCatalogT<RecordT>, std::shared_ptr<SortedCatalogT<RecordT>>, CatalogT<RecordT>> & cls
) {
    using Catalog = SortedCatalogT<RecordT>;
    using Table = typename RecordT::Table;

    /* Constructors */
    cls.def(py::init<Schema const &>());
    cls.def(py::init<PTR(Table) const &>(), "table"_a=PTR(Table)());
    cls.def(py::init<Catalog const &>());

    /* Overridden and Variant Methods */
    cls.def_static("readFits",
                   (Catalog (*)(std::string const &, int, int)) &Catalog::readFits,
                   "filename"_a, "hdu"_a=0, "flags"_a=0);
    cls.def_static("readFits",
                   (Catalog (*)(fits::MemFileManager &, int, int)) &Catalog::readFits,
                   "manager"_a, "hdu"_a=0, "flags"_a=0);
    cls.def("subset",
            (Catalog (Catalog::*)(ndarray::Array<bool const,1> const &) const) &Catalog::subset);
    cls.def("subset",
            (Catalog (Catalog::*)(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t) const) &Catalog::subset);
};

}}}} // lsst::afw::table::pybind11

#endif // !AFW_TABLE_PYBIND11_CATALOG_H_INCLUDED
