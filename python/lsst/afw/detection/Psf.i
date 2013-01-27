// -*- lsst-C++ -*-

%{
#include "lsst/afw/detection/Psf.h"
%}

%import "lsst/afw/table/io/Persistable.i"

%declareTablePersistable(Psf, lsst::afw::detection::Psf);
%declareTablePersistable(KernelPsf, lsst::afw::detection::KernelPsf);

%include "lsst/afw/detection/Psf.h"

%lsst_persistable(lsst::afw::detection::Psf);
%lsst_persistable(lsst::afw::detection::KernelPsf);
