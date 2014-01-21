// -*- lsst-C++ -*-

%{
#include "lsst/afw/detection/GaussianPsf.h"
%}

%import "lsst/afw/table/io/ioLib.i"

%declareTablePersistable(GaussianPsf, lsst::afw::detection::GaussianPsf);

%include "lsst/afw/detection/GaussianPsf.h"

%lsst_persistable(lsst::afw::detection::GaussianPsf);
