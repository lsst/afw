%define fitsLib_DOCSTRING
"
Python interface to lsst::afw::fits exception classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.fits", docstring=fitsLib_DOCSTRING) fitsLib

%{
#include "lsst/afw/fits.h"
#include "lsst/pex/exceptions.h"
%}

%include "cdata.i"

%include "lsst/p_lsstSwig.i"

%lsst_exceptions();

%pythonnondynamic;
%naturalvar;  // use const reference typemaps

%import "lsst/pex/exceptions/exceptionsLib.i"

%include "lsst/afw/fits.h"

%declareException(FitsError, lsst.pex.exceptions.IoError, lsst::afw::fits::FitsError)
%declareException(FitsTypeError, FitsError, lsst::afw::fits::FitsTypeError)

namespace lsst { namespace afw { namespace fits {

struct MemFileManager {
     MemFileManager();
     MemFileManager(std::size_t len);
     void* getData() const;
     std::size_t getLength() const;
};

}}}
