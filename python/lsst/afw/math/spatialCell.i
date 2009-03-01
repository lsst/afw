%{
#include "lsst/afw/math/SpatialCell.h"
%}

SWIG_SHARED_PTR(SpatialCellCandidate, lsst::afw::math::SpatialCellCandidate);
SWIG_SHARED_PTR(SpatialCell, lsst::afw::math::SpatialCell);

%include "lsst/afw/math/SpatialCell.h"

%template(SpatialCellCandidateList) std::vector<lsst::afw::math::SpatialCellCandidate::Ptr>;
%template(SpatialCellList) std::vector<lsst::afw::math::SpatialCell::Ptr>;
