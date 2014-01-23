// -*- lsst-C++ -*-

%{
#include "lsst/afw/detection/GaussianPsf.h"
%}

%import "lsst/afw/table/io/ioLib.i"

%declareTablePersistable(GaussianPsf, lsst::afw::detection::GaussianPsf);

%include "lsst/afw/detection/GaussianPsf.h"

%castShared(lsst::afw::detection::GaussianPsf, lsst::afw::detection::Psf)
