// -*- lsst-C++ -*-
//
// A couple of macros (%IMAGE and %MASKEDIMAGE) to provide MaskedImage's default arguments,
// We'll use these to define meta-macros (e.g. %SpatialCellImageCandidatePtr)
//
%define %IMAGE(PIXTYPE)
lsst::afw::image::Image<PIXTYPE>
%enddef

%define %MASKEDIMAGE(PIXTYPE)
lsst::afw::image::MaskedImage<PIXTYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>
%enddef

//
// Must go BEFORE the include
//
%define %SpatialCellImageCandidatePtr_(NAME, TYPE, ...)
SWIG_SHARED_PTR_DERIVED(SpatialCellImageCandidate##NAME,
                        lsst::afw::math::SpatialCellCandidate,
                        lsst::afw::math::SpatialCellImageCandidate<TYPE >);
%enddef

%define %SpatialCellImageCandidatePtr(NAME, TYPE, ...)
   %SpatialCellImageCandidatePtr_(NAME, %IMAGE(TYPE))
   %SpatialCellImageCandidatePtr_(M##NAME, %MASKEDIMAGE(TYPE))
%enddef
//
// Must go AFTER the include
//
%define %SpatialCellImageCandidate_(NAME, TYPE, ...)
%template(SpatialCellCandidateImage##NAME) lsst::afw::math::SpatialCellImageCandidate<TYPE >;

%inline %{
    lsst::afw::math::SpatialCellImageCandidate<TYPE > *
        cast_SpatialCellImageCandidate##NAME(lsst::afw::math::SpatialCellCandidate* candidate) {
        return dynamic_cast<lsst::afw::math::SpatialCellImageCandidate<TYPE > *>(candidate);
    }
%}
%enddef

%define %SpatialCellImageCandidate(NAME, TYPE, ...)
   %SpatialCellImageCandidate_(NAME, %IMAGE(TYPE))
   %SpatialCellImageCandidate_(M##NAME, %MASKEDIMAGE(TYPE))
%enddef

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
// OK, now we'll generate some code
//

%{
#include "lsst/afw/math/SpatialCell.h"
%}

SWIG_SHARED_PTR(SpatialCellCandidate, lsst::afw::math::SpatialCellCandidate);
SWIG_SHARED_PTR(SpatialCell, lsst::afw::math::SpatialCell);

%SpatialCellImageCandidatePtr(F, float);

%include "lsst/afw/math/SpatialCell.h"

%template(SpatialCellCandidateList) std::vector<lsst::afw::math::SpatialCellCandidate::Ptr>;
%template(SpatialCellList) std::vector<lsst::afw::math::SpatialCell::Ptr>;

%SpatialCellImageCandidate(F, float);
