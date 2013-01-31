// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelPersistenceHelper.h"

namespace lsst { namespace afw { namespace math {

Kernel::PersistenceHelper::PersistenceHelper(int nSpatialFunctions) :
    schema(),
    dimensions(schema.addField< afw::table::Point<int> >("dimensions", "dimensions of a Kernel's images")),
    center(schema.addField< afw::table::Point<int> >("center", "center point in a Kernel image"))
{
    if (nSpatialFunctions > 0) {
        spatialFunctions = schema.addField< afw::table::Array<int> >(
            "spatialfunctions", "archive IDs for the Kernel's spatial functions", nSpatialFunctions
        );
    }
}

Kernel::PersistenceHelper::PersistenceHelper(afw::table::Schema const & schema_) :
    schema(schema_),
    dimensions(schema["dimensions"]),
    center(schema["center"])
{
    try {
        spatialFunctions = schema["spatialfunctions"];
    } catch (...) {}
}

PTR(afw::table::BaseRecord) Kernel::PersistenceHelper::write(
    afw::table::io::OutputArchiveHandle & handle,
    Kernel const & kernel
) const {
    afw::table::BaseCatalog catalog = handle.makeCatalog(schema);
    PTR(afw::table::BaseRecord) record = catalog.addNew();
    record->set(dimensions, geom::Point2I(kernel.getDimensions()));
    record->set(center, kernel.getCtr());
    if (spatialFunctions.isValid()) {
        writeSpatialFunctions(handle, *record, kernel._spatialFunctionList);
    }
    handle.saveCatalog(catalog);
    return record;
}

void Kernel::PersistenceHelper::writeSpatialFunctions(
    afw::table::io::OutputArchiveHandle & handle,
    afw::table::BaseRecord & record,
    std::vector<PTR(Kernel::SpatialFunction)> const & spatialFunctionList
) const {
    ndarray::Array<int,1,1> array = record[spatialFunctions];
    for (std::size_t n = 0; n < spatialFunctionList.size(); ++n) {
        array[n] = handle.put(spatialFunctionList[n]);
    }
}

std::vector<PTR(Kernel::SpatialFunction)> Kernel::PersistenceHelper::readSpatialFunctions(
    afw::table::io::InputArchive const & archive,
    afw::table::BaseRecord const & record
) const {
    ndarray::Array<int const,1,1> array = record[spatialFunctions];
    std::vector<PTR(Kernel::SpatialFunction)> spatialFunctionList(array.getSize<0>());
    for (std::size_t n = 0; n < spatialFunctionList.size(); ++n) {
        spatialFunctionList[n] = archive.get<SpatialFunction>(array[n]);
        LSST_ARCHIVE_ASSERT(array[n] == 0 || (spatialFunctionList[n]));
    }
    return spatialFunctionList;
}

}}} // namespace lsst::afw::math
