// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelPersistenceHelper.h"

namespace lsst { namespace afw { namespace math {

Kernel::PersistenceHelper::PersistenceHelper(int nSpatialFunctions) :
    schema(),
    dimensions(
        afw::table::PointKey<int>::addFields(
            schema, "dimensions", "dimensions of a Kernel's images", "pixels"
        )
    ),
    center(
        afw::table::PointKey<int>::addFields(schema, "center", "center point in a Kernel image", "pixels")
    )
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
