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
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseColumnView.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/pybind11/catalog.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace table {

PYBIND11_PLUGIN(_baseCatalog) {
    py::module mod("_baseCatalog", "Python wrapper for afw _baseCatalog library");

    typedef CatalogT<BaseRecord> BaseCatalog;

    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
    };

    /* Module level */
    py::class_<BaseCatalog, std::shared_ptr<BaseCatalog>> clsCatalog(mod, "BaseCatalog");

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */
    declareCatalog<BaseRecord>(clsCatalog);

    return mod.ptr();
}

}}}  // namespace lsst::afw::table
