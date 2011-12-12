// -*- lsst-C++ -*-

%{
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/detection/LocalPsf.h"
%}

SWIG_SHARED_PTR_DERIVED(PSFPtrT, lsst::daf::base::Citizen, lsst::afw::detection::Psf);
SWIG_SHARED_PTR_DERIVED(PSFPtrT, lsst::afw::detection::Psf, lsst::afw::detection::KernelPsf);

%declareNumPyConverters(lsst::ndarray::Array<lsst::afw::detection::LocalPsf::Pixel,1,0>);
%declareNumPyConverters(lsst::ndarray::Array<lsst::afw::detection::LocalPsf::Pixel const,1,0>);

%ignore PsfFactoryBase;
%include "lsst/afw/detection/Psf.h"

%lsst_persistable(lsst::afw::detection::Psf);

SWIG_SHARED_PTR(LocalPsfPtrT, lsst::afw::detection::LocalPsf);
SWIG_SHARED_PTR_DERIVED(ImageLocalPsfPtrT, lsst::afw::detection::LocalPsf, 
                        lsst::afw::detection::ImageLocalPsf);
SWIG_SHARED_PTR_DERIVED(ShapeletLocalPsfPtrT, lsst::afw::detection::LocalPsf, 
                        lsst::afw::detection::ShapeletLocalPsf);

%include "lsst/afw/detection/LocalPsf.h"
