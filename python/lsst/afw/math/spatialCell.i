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
// Must go AFTER the include (N.b. %SpatialCellImageCandidate_ is internal)
//
%define %SpatialCellImageCandidate_(NAME, TYPE)
%template(SpatialCellImageCandidate##NAME) lsst::afw::math::SpatialCellImageCandidate<TYPE>;

//
// When swig sees a SpatialCellCandidate it doesn't know about SpatialCellImageCandidates; all it knows is that it
// has a SpatialCellCandidate, and SpatialCellCandidates don't know about e.g. getSource()
//
// We therefore provide a cast to SpatialCellImageCandidate<> and swig can go from there
//
%inline %{
    lsst::afw::math::SpatialCellImageCandidate<TYPE> *
        cast_SpatialCellImageCandidate##NAME(lsst::afw::math::SpatialCellCandidate* candidate) {
        return dynamic_cast<lsst::afw::math::SpatialCellImageCandidate<TYPE> *>(candidate);
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

%rename(__incr__) lsst::afw::math::SpatialCellCandidateIterator::operator++;
%rename(__deref__) lsst::afw::math::SpatialCellCandidateIterator::operator*;
%rename(__eq__) lsst::afw::math::SpatialCellCandidateIterator::operator==;
%rename(__ne__) lsst::afw::math::SpatialCellCandidateIterator::operator!=;

%include "lsst/afw/math/SpatialCell.h"

%template(SpatialCellCandidateList) std::vector<lsst::afw::math::SpatialCellCandidate::Ptr>;
%template(SpatialCellList) std::vector<lsst::afw::math::SpatialCell::Ptr>;

%SpatialCellImageCandidate(F, float);


%extend lsst::afw::math::SpatialCell {
    %pythoncode {
        def __getitem__(self, ind):
            return [c for c in self.begin()][ind]
    }
}

%extend lsst::afw::math::SpatialCellCandidateIterator {
    %pythoncode {
    def __iter__(self):
        while True:
            try:
                yield self.__deref__()
            except:
                raise StopIteration
            
            self.__incr__()
    }
}
