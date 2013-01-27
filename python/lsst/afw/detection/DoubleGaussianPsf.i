// -*- lsst-C++ -*-

%include "lsst/afw/detection/Psf.i"

%{
#include "lsst/afw/detection/DoubleGaussianPsf.h"
%}

%import "lsst/afw/table/io/Persistable.i"

%declareTablePersistable(DoubleGaussianPsf, lsst::afw::detection::DoubleGaussianPsf);

%include "lsst/afw/detection/DoubleGaussianPsf.h"

%lsst_persistable(lsst::afw::detection::DoubleGaussianPsf);
