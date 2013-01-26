// -*- lsst-C++ -*-

%{
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/detection/DoubleGaussianPsf.h"
%}

%import "lsst/afw/table/io/Persistable.i"

%declareTablePersistable(Psf, lsst::afw::detection::Psf);
%declareTablePersistable(KernelPsf, lsst::afw::detection::KernelPsf);
%declareTablePersistable(DoubleGaussianPsf, lsst::afw::detection::DoubleGaussianPsf);

%ignore PsfFactoryBase;
%include "lsst/afw/detection/Psf.h"
%include "lsst/afw/detection/DoubleGaussianPsf.h"

%lsst_persistable(lsst::afw::detection::Psf);
%lsst_persistable(lsst::afw::detection::DoubleGaussianPsf);
