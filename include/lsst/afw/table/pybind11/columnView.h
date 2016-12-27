#ifndef AFW_TABLE_PYBIND11_COLUMNVIEW_H_INCLUDED
#define AFW_TABLE_PYBIND11_COLUMNVIEW_H_INCLUDED
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

#include "lsst/afw/table/BaseColumnView.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace table {
namespace pybind11 {

/**
Declare member and static functions for a given instantiation of lsst::afw::table::ColumnViewT<RecordT>.

@tparam RecordT  Record type, e.g. BaseRecord or SimpleRecord.

@param[in] cls  Catalog pybind11 class.
*/
template <typename RecordT>
void declareColumnView(py::class_<ColumnViewT<RecordT>,
                       std::shared_ptr<ColumnViewT<RecordT>>,
                       BaseColumnView> & cls) {
    cls.def("getTable", &ColumnViewT<RecordT>::getTable);
    cls.def_property_readonly("table", &ColumnViewT<RecordT>::getTable);
};

}}}}  // namespace lsst::afw::table::pybind11
#endif
