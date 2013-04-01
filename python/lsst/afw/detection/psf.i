// -*- lsst-C++ -*-

%{
#include "lsst/afw/detection/Psf.h"
%}

%import "lsst/afw/table/io/ioLib.i"

%declareTablePersistable(Psf, lsst::afw::detection::Psf);

%include "lsst/afw/detection/Psf.h"

%lsst_persistable(lsst::afw::detection::Psf);
