

%{
#include "lsst/afw/detection/FootprintSet.h"
#include "lsst/afw/table/Source.h"
%}

%import "lsst/afw/table/Source.i"

%shared_ptr(lsst::afw::detection::FootprintSet);

%include "lsst/afw/detection/FootprintSet.h"

%define %footprintSetOperations(PIXEL)
// uncomment the following two lines and update FootprintSet.h accordingly
// once https://github.com/swig/swig/issues/245 is fixed
// %template(FootprintSet) FootprintSet<PIXEL>;
// %template(FootprintSet) FootprintSet<PIXEL,lsst::afw::image::MaskPixel>;
%template(makeHeavy) makeHeavy<PIXEL,lsst::afw::image::MaskPixel>;
%template(setMask) setMask<lsst::afw::image::MaskPixel>;
%enddef

%extend lsst::afw::detection::FootprintSet {
%footprintSetOperations(boost::uint16_t)
%footprintSetOperations(int)
%footprintSetOperations(float)
%footprintSetOperations(double)
}

namespace lsst { namespace afw { namespace table {
     typedef VectorT< lsst::afw::table::SourceRecord, lsst::afw::table::SourceTable > SourceVector;
}}}


