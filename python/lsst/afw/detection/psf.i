// -*- lsst-C++ -*-

%{
#include "lsst/afw/detection/Psf.h"
%}

SWIG_SHARED_PTR_DERIVED(PSFPtrT, lsst::daf::data::LsstBase, lsst::afw::detection::Psf);
SWIG_SHARED_PTR_DERIVED(PSFPtrT, lsst::afw::detection::Psf, lsst::afw::detection::KernelPsf);

%ignore PsfFactoryBase;
%include "lsst/afw/detection/Psf.h"

%lsst_persistable(lsst::afw::detection::Psf);
