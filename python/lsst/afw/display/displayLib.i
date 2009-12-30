// -*- lsst-c++ -*-
%define displayLib_DOCSTRING
"
Basic routines to talk to ds9
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.display", docstring=displayLib_DOCSTRING) displayLib

%{
#   include "lsst/daf/base.h"
#   include "lsst/daf/data.h"
#   include "lsst/daf/persistence.h"
#   include "lsst/pex/policy.h"
#   include "lsst/pex/logging/Log.h"
#   include "lsst/afw/geom.h"
#   include "lsst/afw/image.h"

#   include "simpleFits.h"
%}

%include "lsst/p_lsstSwig.i"

%import "lsst/afw/image/imageLib.i"

%lsst_exceptions();

%include "simpleFits.h"

%template(writeFitsImage) lsst::afw::display::writeBasicFits<lsst::afw::image::Image<boost::uint16_t> >;
%template(writeFitsImage) lsst::afw::display::writeBasicFits<lsst::afw::image::Image<int> >;
%template(writeFitsImage) lsst::afw::display::writeBasicFits<lsst::afw::image::Image<float> >;
%template(writeFitsImage) lsst::afw::display::writeBasicFits<lsst::afw::image::Image<double> >;
%template(writeFitsImage) lsst::afw::display::writeBasicFits<lsst::afw::image::Mask<boost::uint16_t> >;

