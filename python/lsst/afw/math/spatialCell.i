%{
#include "lsst/afw/math/SpatialCell.h"
%}

SWIG_SHARED_PTR(SpatialCellCandidate, lsst::afw::math::SpatialCellCandidate);

%include "lsst/afw/math/SpatialCell.h"

%template(SpatialCellCandidateList) std::vector<lsst::afw::math::SpatialCellCandidate::Ptr>;
