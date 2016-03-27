// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#ifndef LSST_AFW_MATH_KernelPersistenceHelper_h_INCLUDED
#define LSST_AFW_MATH_KernelPersistenceHelper_h_INCLUDED

#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/aggregates.h"

namespace lsst { namespace afw { namespace math {

// Schema for use by Kernel subclasses in persistence.
struct Kernel::PersistenceHelper {
    afw::table::Schema schema;
    afw::table::PointKey<int> dimensions;
    afw::table::PointKey<int> center;
    afw::table::Key< afw::table::Array<int> > spatialFunctions;

    explicit PersistenceHelper(int nSpatialFunctions);
    explicit PersistenceHelper(afw::table::Schema const & schema_);

    PTR(afw::table::BaseRecord) write(
        afw::table::io::OutputArchiveHandle & handle,
        Kernel const & kernel
    ) const;

    void writeSpatialFunctions(
        afw::table::io::OutputArchiveHandle & handle,
        afw::table::BaseRecord & record,
        std::vector<SpatialFunctionPtr> const & spatialFunctionList
    ) const;

    std::vector<SpatialFunctionPtr> readSpatialFunctions(
        afw::table::io::InputArchive const & archive,
        afw::table::BaseRecord const & record
    ) const;
};

}}}   // lsst:afw::math

#endif // !LSST_AFW_MATH_KernelPersistenceHelper_h_INCLUDED)
