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

#ifndef LSST_AFW_MATH_KernelPersistenceHelper_h_INCLUDED
#define LSST_AFW_MATH_KernelPersistenceHelper_h_INCLUDED

#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/aggregates.h"

namespace lsst {
namespace afw {
namespace math {

// Schema for use by Kernel subclasses in persistence.
struct Kernel::PersistenceHelper {
    afw::table::Schema schema;
    afw::table::PointKey<int> dimensions;
    afw::table::PointKey<int> center;
    afw::table::Key<afw::table::Array<int> > spatialFunctions;

    explicit PersistenceHelper(int nSpatialFunctions);
    explicit PersistenceHelper(afw::table::Schema const& schema_);

    std::shared_ptr<afw::table::BaseRecord> write(afw::table::io::OutputArchiveHandle& handle,
                                                  Kernel const& kernel) const;

    void writeSpatialFunctions(afw::table::io::OutputArchiveHandle& handle, afw::table::BaseRecord& record,
                               std::vector<SpatialFunctionPtr> const& spatialFunctionList) const;

    std::vector<SpatialFunctionPtr> readSpatialFunctions(afw::table::io::InputArchive const& archive,
                                                         afw::table::BaseRecord const& record) const;
};
}
}
}  // lsst:afw::math

#endif  // !LSST_AFW_MATH_KernelPersistenceHelper_h_INCLUDED)
