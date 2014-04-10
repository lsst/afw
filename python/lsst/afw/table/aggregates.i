/*
 * LSST Data Management System
 * Copyright 2008-2014 LSST Corporation.
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
 * Wrappers for FunctorKeys that map objects to groups of fields
 */

%include "lsst/afw/table/Base.i"

%{
#include "lsst/afw/table/aggregates.h"
%}

// =============== PointKey =================================================================================

%declareFunctorKey(Point2I, lsst::afw::geom::Point<int,2>)
%declareFunctorKey(Point2D, lsst::afw::geom::Point<double,2>)
%shared_ptr(lsst::afw::table::PointKey<int>)
%shared_ptr(lsst::afw::table::PointKey<double>)

// =============== QuadrupoleKey ============================================================================

%declareFunctorKey(Quadrupole, lsst::afw::geom::ellipses::Quadrupole)
%shared_ptr(lsst::afw::table::QuadrupoleKey)

// =============== Include and Template Instantiation =======================================================

%include "lsst/afw/table/aggregates.h"

%template(Point2IKey) lsst::afw::table::PointKey<int>;
%template(Point2DKey) lsst::afw::table::PointKey<double>;


%useValueEquality(lsst::afw::table::PointKey<int>)
%useValueEquality(lsst::afw::table::PointKey<double>)
%useValueEquality(lsst::afw::table::QuadrupoleKey)
