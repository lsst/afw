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

%define %declarePointKey1(SUFFIX, T)  // invoke before %include
%declareFunctorKey(Point2 ## SUFFIX, lsst::afw::geom::Point<T,2>)
%shared_ptr(lsst::afw::table::PointKey<T>);
%enddef

%define %declarePointKey2(SUFFIX, T)  // invoke after %include
%template(Point2 ## SUFFIX ## Key) lsst::afw::table::PointKey<T>;
%useValueEquality(lsst::afw::table::PointKey<T>)
%enddef

%declarePointKey1(I, int)
%declarePointKey1(D, double)

// =============== CoordKey==================================================================================

%declareFunctorKey(Coord, lsst::afw::coord::Coord)
%shared_ptr(lsst::afw::table::CoordKey)
%useValueEquality(lsst::afw::table::CoordKey)

// =============== QuadrupoleKey ============================================================================

%declareFunctorKey(Quadrupole, lsst::afw::geom::ellipses::Quadrupole)
%shared_ptr(lsst::afw::table::QuadrupoleKey)

// =============== CovarianceMatrixKey ======================================================================

%template(KeyFVector) std::vector< lsst::afw::table::Key<float> >;
%template(KeyDVector) std::vector< lsst::afw::table::Key<double> >;

%define %declareCovarianceMatrixKey1(SUFFIX, N, T)
%declareFunctorKey(Matrix ## SUFFIX, Eigen::Matrix<T,N,N>)
%shared_ptr(lsst::afw::table::CovarianceMatrixKey<T,N>)
%declareNumPyConverters(Eigen::Matrix<T,N,N>);
%enddef

%define %declareCovarianceMatrixKey2(SUFFIX, N, T)
%template(CovarianceMatrix ## SUFFIX ## Key) lsst::afw::table::CovarianceMatrixKey<T,N>;
%useValueEquality(lsst::afw::table::CovarianceMatrixKey<T,N>)
%enddef

%declareCovarianceMatrixKey1(2f, 2, float)
%declareCovarianceMatrixKey1(3f, 3, float)
%declareCovarianceMatrixKey1(4f, 4, float)
%declareCovarianceMatrixKey1(Xf, Eigen::Dynamic, float)

%declareCovarianceMatrixKey1(2d, 2, double)
%declareCovarianceMatrixKey1(3d, 3, double)
%declareCovarianceMatrixKey1(4d, 4, double)
%declareCovarianceMatrixKey1(Xd, Eigen::Dynamic, double)

// =============== Include and Template Instantiation =======================================================

%include "lsst/afw/table/aggregates.h"

%declarePointKey2(I, int)
%declarePointKey2(D, double)
%useValueEquality(lsst::afw::table::QuadrupoleKey)

%declareCovarianceMatrixKey2(2f, 2, float)
%declareCovarianceMatrixKey2(3f, 3, float)
%declareCovarianceMatrixKey2(4f, 4, float)
%declareCovarianceMatrixKey2(Xf, Eigen::Dynamic, float)

%declareCovarianceMatrixKey2(2d, 2, double)
%declareCovarianceMatrixKey2(3d, 3, double)
%declareCovarianceMatrixKey2(4d, 4, double)
%declareCovarianceMatrixKey2(Xd, Eigen::Dynamic, double)
