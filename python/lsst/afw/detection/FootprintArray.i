// -*- lsst-c++ -*-

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

%include "lsst/afw/detection/detection_fwd.i"
%include "lsst/afw/detection/Footprint.i"

%{
#include "lsst/afw/detection/FootprintArray.h"
#include "lsst/afw/detection/FootprintArray.cc"
%}

// because stupid SWIG's %template doesn't work on these functions
%define %footprintArrayTemplates(T)
%declareNumPyConverters(ndarray::Array<T,1,0>);
%declareNumPyConverters(ndarray::Array<T,2,0>);
%declareNumPyConverters(ndarray::Array<T,3,0>);
%declareNumPyConverters(ndarray::Array<T const,1,0>);
%declareNumPyConverters(ndarray::Array<T const,2,0>);
%declareNumPyConverters(ndarray::Array<T const,3,0>);
%inline %{
    void flattenArray(
        lsst::afw::detection::Footprint const & fp,
        ndarray::Array<T const,2,0> const & src,
        ndarray::Array<T,1,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::flattenArray(fp, src, dest, origin);
    }    
    void flattenArray(
        lsst::afw::detection::Footprint const & fp,
        ndarray::Array<T const,3,0> const & src,
        ndarray::Array<T,2,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::flattenArray(fp, src, dest, origin);
    }    
    void expandArray(
        lsst::afw::detection::Footprint const & fp,
        ndarray::Array<T const,1,0> const & src,
        ndarray::Array<T,2,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::expandArray(fp, src, dest, origin);
    }
    void expandArray(
        lsst::afw::detection::Footprint const & fp,
        ndarray::Array<T const,2,0> const & src,
        ndarray::Array<T,3,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::expandArray(fp, src, dest, origin);
    }
%}
%{
    template void lsst::afw::detection::flattenArray(
        lsst::afw::detection::Footprint const &,
        ndarray::Array<T const,2,0> const &,
        ndarray::Array<T,1,0> const &,
        lsst::afw::geom::Point2I const &
    );
    template void lsst::afw::detection::flattenArray(
        lsst::afw::detection::Footprint const &,
        ndarray::Array<T const,3,0> const &,
        ndarray::Array<T,2,0> const &,
        lsst::afw::geom::Point2I const &
    );
    template void lsst::afw::detection::expandArray(
        lsst::afw::detection::Footprint const &,
        ndarray::Array<T const,1,0> const &,
        ndarray::Array<T,2,0> const &,
        lsst::afw::geom::Point2I const &
    );
    template void lsst::afw::detection::expandArray(
        lsst::afw::detection::Footprint const &,
        ndarray::Array<T const,2,0> const &,
        ndarray::Array<T,3,0> const &,
        lsst::afw::geom::Point2I const &
    );
%}
%enddef

%footprintArrayTemplates(boost::uint16_t);
%footprintArrayTemplates(int);
%footprintArrayTemplates(float);
%footprintArrayTemplates(double);

