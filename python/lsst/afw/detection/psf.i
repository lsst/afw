// -*- lsst-C++ -*-

%{
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/detection/LocalPsf.h"
%}

%shared_ptr(lsst::afw::detection::Psf);
%shared_ptr(lsst::afw::detection::KernelPsf);

%declareNumPyConverters(lsst::ndarray::Array<lsst::afw::detection::LocalPsf::Pixel,1,0>);
%declareNumPyConverters(lsst::ndarray::Array<lsst::afw::detection::LocalPsf::Pixel const,1,0>);

%ignore PsfFactoryBase;
%include "lsst/afw/detection/Psf.h"

%lsst_persistable(lsst::afw::detection::Psf);

%shared_ptr(lsst::afw::detection::LocalPsf);
%shared_ptr(lsst::afw::detection::ImageLocalPsf);
%shared_ptr(lsst::afw::detection::ShapeletLocalPsf);

%include "lsst/afw/detection/LocalPsf.h"
