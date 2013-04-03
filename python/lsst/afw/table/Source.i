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
 * Wrappers for SourceTable, SourceRecord, and SourceColumnView (and, via Simple.i, their dependencies).
 * Also includes SortedCatalog.i and instantiates SourceCatalog.
 *
 * This file does not include Exposure- Record/Table/Catalog, or the matching functions.
 */

%include "lsst/afw/table/Simple.i"

%{
#include "lsst/afw/table/Source.h"
%}

namespace lsst { namespace afw { namespace image {
class Wcs;
}
namespace detection {
class Footprint;
}
}}
%shared_ptr(lsst::afw::image::Wcs);
%shared_ptr(lsst::afw::detection::Footprint);

// =============== SourceTable and SourceRecord =============================================================

%shared_ptr(lsst::afw::table::SourceTable)
%shared_ptr(lsst::afw::table::SourceRecord)
// Workarounds for SWIG's failure to parse the Measurement template correctly.
// Otherwise we'd have one place in the code that controls all the canonical measurement types.
namespace lsst { namespace afw { namespace table {
     struct Flux {
         typedef Key< double > MeasKey;
         typedef Key< double > ErrKey;
         typedef double MeasValue;
         typedef double ErrValue;
     };
     struct Centroid {
         typedef Key< Point<double> > MeasKey;
         typedef Key< Covariance< Point<float> > > ErrKey;
         typedef lsst::afw::geom::Point<double,2> MeasValue;
         typedef Eigen::Matrix<float,2,2> ErrValue;
     };
     struct Shape {
         typedef Key< Moments<double> > MeasKey;
         typedef Key< Covariance< Moments<float> > > ErrKey;
         typedef lsst::afw::geom::ellipses::Quadrupole MeasValue;
         typedef Eigen::Matrix<float,3,3> ErrValue;
     };
}}}

%define %enableSlotKwArgs(SLOT)
%extend lsst::afw::table::SourceTable {
    %feature("shadow") define ## SLOT %{
    def define ## SLOT(self, meas, err=None, flag=None):
        if err is None:
            if flag is None:
                $action(self, meas)
            else:
                $action(self, meas, flag)
        else:
            if flag is None:
                $action(self, meas, err)
            else:
                $action(self, meas, err, flag)
    %}
}
%enddef

%enableSlotKwArgs(PsfFlux)
%enableSlotKwArgs(ApFlux)
%enableSlotKwArgs(InstFlux)
%enableSlotKwArgs(ModelFlux)
%enableSlotKwArgs(Centroid)
%enableSlotKwArgs(Shape)

%include "lsst/afw/table/Source.h"

%addCastMethod(lsst::afw::table::SourceTable, lsst::afw::table::BaseTable)
%addCastMethod(lsst::afw::table::SourceRecord, lsst::afw::table::BaseRecord)

%addCastMethod(lsst::afw::table::SourceTable, lsst::afw::table::SimpleTable)
%addCastMethod(lsst::afw::table::SourceRecord, lsst::afw::table::SimpleRecord)

%template(_SourceColumnViewBase) lsst::afw::table::ColumnViewT<lsst::afw::table::SourceRecord>;
%template(SourceColumnView) lsst::afw::table::SourceColumnViewT<lsst::afw::table::SourceRecord>;

// =============== Catalogs =================================================================================

%include "lsst/afw/table/SortedCatalog.i"

namespace lsst { namespace afw { namespace table {

%declareSortedCatalog(SortedCatalogT, Source)

}}} // namespace lsst::afw::table
