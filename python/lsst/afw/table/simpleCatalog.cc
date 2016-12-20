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
#include "lsst/afw/table/pybind11/catalog.h"
#include "lsst/afw/table/pybind11/sortedCatalog.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace table {

PYBIND11_PLUGIN(_simpleCatalog) {
    py::module mod("_simpleCatalog", "Python wrapper for afw _simpleCatalog library");

    typedef py::class_<CatalogT<SimpleRecord>, std::shared_ptr<CatalogT<SimpleRecord>>> PyBaseSimpleCatalog;

    typedef py::class_<SimpleCatalog, std::shared_ptr<SimpleCatalog>, CatalogT<SimpleRecord>> PySimpleCatalog;

    /* Module level */
    PyBaseSimpleCatalog clsBaseSimpleCatalog(mod, "_BaseSimpleCatalog");

    PySimpleCatalog clsSimpleCatalog(mod, "SimpleCatalog");

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */
    declareCatalog<SimpleRecord>(clsBaseSimpleCatalog);

    declareSortedCatalog<SimpleRecord>(clsSimpleCatalog);

    return mod.ptr();
}

}}} // lsst::afw::table
