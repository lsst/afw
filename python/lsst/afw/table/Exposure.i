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

/*
 * Wrappers for ExposureTable, ExposureRecord, and ExposureCatalog (and, via Base.i, their dependencies).
 *
 * This file does not include Simple- or Source- Record/Table/Catalog, or the matching functions.
 */

%include "lsst/afw/table/Base.i"

%{
#include "lsst/afw/table/Exposure.h"
%}

namespace lsst { namespace afw { namespace image {
class Wcs;
class Calib;
}
namespace detection {
class Psf;
}
}}
%shared_ptr(lsst::afw::image::Wcs);
%shared_ptr(lsst::afw::image::Calib);
%shared_ptr(lsst::afw::image::ApCorrMap);
%shared_ptr(lsst::afw::detection::Psf);
%shared_ptr(lsst::afw::geom::polygon::Polygon)

// =============== ExposureTable and ExposureRecord =========================================================

%shared_ptr(lsst::afw::table::ExposureTable)
%shared_ptr(lsst::afw::table::ExposureRecord)

%include "lsst/afw/table/Exposure.h"

%addCastMethod(lsst::afw::table::ExposureTable, lsst::afw::table::BaseTable)
%addCastMethod(lsst::afw::table::ExposureRecord, lsst::afw::table::BaseRecord)

%template(ExposureColumnView) lsst::afw::table::ColumnViewT<lsst::afw::table::ExposureRecord>;

// =============== Catalogs =================================================================================

%include "lsst/afw/table/SortedCatalog.i"

namespace lsst { namespace afw { namespace table {

template <typename RecordT>
class ExposureCatalogT : public SortedCatalogT<RecordT> {
public:

    typedef typename RecordT::Table Table;

    explicit ExposureCatalogT(PTR(Table) const & table);

    explicit ExposureCatalogT(Schema const & table);

    ExposureCatalogT(ExposureCatalogT const & other);

    %feature(
        "autodoc",
        "Constructors:  __init__(self, table) -> empty catalog with the given table\n"
        "               __init__(self, schema) -> empty catalog with a new table with the given schema\n"
        "               __init__(self, catalog) -> shallow copy of the given catalog\n"
    ) SortedCatalogT;

    static ExposureCatalogT readFits(std::string const & filename, int hdu=0, int flags=0);
    static ExposureCatalogT readFits(fits::MemFileManager & manager, int hdu=0, int flags=0);

    ExposureCatalogT<RecordT> subset(ndarray::Array<bool const,1> const & mask) const;
    ExposureCatalogT<RecordT> subset(std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step) const;

    ExposureCatalogT subsetContaining(Coord const & coord, bool includeValidPolygon=false) const;
    ExposureCatalogT subsetContaining(geom::Point2D const & point, image::Wcs const & wcs,
                                      bool includeValidPolygon=false) const;
};

%pythondynamic;
%template (_ExposureCatalogBase) CatalogT<ExposureRecord>;
%template (_ExposureCatalogSortedBase) SortedCatalogT<ExposureRecord>;
%pythonnondynamic;
%declareCatalog(ExposureCatalogT, Exposure)

}}} // namespace lsst::afw::table
