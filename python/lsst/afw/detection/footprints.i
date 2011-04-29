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
 
%ignore lsst::afw::detection::FootprintFunctor::operator();

%{
#include "lsst/afw/detection/Threshold.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/detection/FootprintSet.h"
#include "lsst/afw/detection/FootprintFunctor.h"
#include "lsst/afw/detection/FootprintArray.h"
#include "lsst/afw/detection/FootprintArray.cc"
%}

// already in image.i.
// %template(VectorBox2I) std::vector<lsst::afw::geom::Box2I>;

SWIG_SHARED_PTR(Peak,      lsst::afw::detection::Peak);
SWIG_SHARED_PTR(Footprint, lsst::afw::detection::Footprint);
SWIG_SHARED_PTR(Span,      lsst::afw::detection::Span);
SWIG_SHARED_PTR(FootprintSetU, lsst::afw::detection::FootprintSet<boost::uint16_t, lsst::afw::image::MaskPixel>);
SWIG_SHARED_PTR(FootprintSetI, lsst::afw::detection::FootprintSet<int, lsst::afw::image::MaskPixel>);
SWIG_SHARED_PTR(FootprintSetF, lsst::afw::detection::FootprintSet<float, lsst::afw::image::MaskPixel>);
SWIG_SHARED_PTR(FootprintSetD, lsst::afw::detection::FootprintSet<double, lsst::afw::image::MaskPixel>);
SWIG_SHARED_PTR(FootprintList, std::vector<lsst::afw::detection::Footprint::Ptr >);

%rename(assign) lsst::afw::detection::Footprint::operator=;

%include "lsst/afw/detection/Threshold.h"
%include "lsst/afw/detection/Peak.h"
%include "lsst/afw/detection/Footprint.h"
%include "lsst/afw/detection/FootprintSet.h"
%include "lsst/afw/detection/FootprintFunctor.h"

%template(PeakContainerT)      std::vector<lsst::afw::detection::Peak::Ptr>;
%template(SpanContainerT)      std::vector<lsst::afw::detection::Span::Ptr>;
%template(FootprintContainerT) std::vector<lsst::afw::detection::Footprint::Ptr>;

%define %footprintOperations(NAME, PIXEL_TYPE)
    %template(FootprintFunctor ##NAME) lsst::afw::detection::FootprintFunctor<lsst::afw::image::Image<PIXEL_TYPE> >;
    %template(FootprintFunctorMI ##NAME)
                       lsst::afw::detection::FootprintFunctor<lsst::afw::image::MaskedImage<PIXEL_TYPE> >;
    %template(setImageFromFootprint) lsst::afw::detection::setImageFromFootprint<lsst::afw::image::Image<PIXEL_TYPE> >;
    %template(setImageFromFootprintList)
                       lsst::afw::detection::setImageFromFootprintList<lsst::afw::image::Image<PIXEL_TYPE> >
%enddef

%define %FootprintSet(NAME, PIXEL_TYPE)
%template(FootprintSet##NAME) lsst::afw::detection::FootprintSet<PIXEL_TYPE, lsst::afw::image::MaskPixel>;
%template(makeFootprintSet) lsst::afw::detection::makeFootprintSet<PIXEL_TYPE, lsst::afw::image::MaskPixel>;
%enddef


%footprintOperations(F, float);
%template(FootprintFunctorMaskU) lsst::afw::detection::FootprintFunctor<lsst::afw::image::Mask<boost::uint16_t> >;

%FootprintSet(U, boost::uint16_t);
%FootprintSet(I, int);
%FootprintSet(D, double);
%FootprintSet(F, float);
%template(makeFootprintSet) lsst::afw::detection::makeFootprintSet<lsst::afw::image::MaskPixel>;

//%template(MaskU) lsst::afw::image::Mask<maskPixelType>;
%template(footprintAndMask) lsst::afw::detection::footprintAndMask<lsst::afw::image::MaskPixel>;
%template(setMaskFromFootprint) lsst::afw::detection::setMaskFromFootprint<lsst::afw::image::MaskPixel>;
%template(setMaskFromFootprintList) lsst::afw::detection::setMaskFromFootprintList<lsst::afw::image::MaskPixel>;

%extend lsst::afw::detection::Span {
    %pythoncode {
    def __str__(self):
        """Print this Span"""
        return self.toString()
    }
}

// because stupid SWIG's %template doesn't work on these functions
%define %footprintArrayTemplates(T)
%declareNumPyConverters(lsst::ndarray::Array<T,1,0>);
%declareNumPyConverters(lsst::ndarray::Array<T,2,0>);
%declareNumPyConverters(lsst::ndarray::Array<T,3,0>);
%declareNumPyConverters(lsst::ndarray::Array<T const,1,0>);
%declareNumPyConverters(lsst::ndarray::Array<T const,2,0>);
%declareNumPyConverters(lsst::ndarray::Array<T const,3,0>);
%inline %{
    void flattenArray(
        lsst::afw::detection::Footprint const & fp,
        lsst::ndarray::Array<T const,2,0> const & src,
        lsst::ndarray::Array<T,1,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::flattenArray(fp, src, dest, origin);
    }    
    void flattenArray(
        lsst::afw::detection::Footprint const & fp,
        lsst::ndarray::Array<T const,3,0> const & src,
        lsst::ndarray::Array<T,2,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::flattenArray(fp, src, dest, origin);
    }    
    void expandArray(
        lsst::afw::detection::Footprint const & fp,
        lsst::ndarray::Array<T const,1,0> const & src,
        lsst::ndarray::Array<T,2,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::expandArray(fp, src, dest, origin);
    }
    void expandArray(
        lsst::afw::detection::Footprint const & fp,
        lsst::ndarray::Array<T const,2,0> const & src,
        lsst::ndarray::Array<T,3,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::expandArray(fp, src, dest, origin);
    }
%}
%{
    template void lsst::afw::detection::flattenArray(
        lsst::afw::detection::Footprint const &,
        lsst::ndarray::Array<T const,2,0> const &,
        lsst::ndarray::Array<T,1,0> const &,
        lsst::afw::geom::Point2I const &
    );
    template void lsst::afw::detection::flattenArray(
        lsst::afw::detection::Footprint const &,
        lsst::ndarray::Array<T const,3,0> const &,
        lsst::ndarray::Array<T,2,0> const &,
        lsst::afw::geom::Point2I const &
    );
    template void lsst::afw::detection::expandArray(
        lsst::afw::detection::Footprint const &,
        lsst::ndarray::Array<T const,1,0> const &,
        lsst::ndarray::Array<T,2,0> const &,
        lsst::afw::geom::Point2I const &
    );
    template void lsst::afw::detection::expandArray(
        lsst::afw::detection::Footprint const &,
        lsst::ndarray::Array<T const,2,0> const &,
        lsst::ndarray::Array<T,3,0> const &,
        lsst::afw::geom::Point2I const &
    );
%}
%enddef

%footprintArrayTemplates(boost::uint16_t);
%footprintArrayTemplates(int);
%footprintArrayTemplates(float);
%footprintArrayTemplates(double);
