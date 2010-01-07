// -*- lsst-c++ -*-
%define cameraGeomLib_DOCSTRING
"
Python bindings for classes describing the the geometry of a mosaic camera
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw", docstring=cameraGeomLib_DOCSTRING) cameraGeomLib

%{
#include "lsst/afw/cameraGeom.h"
%}

%include "lsst/p_lsstSwig.i"
%import  "lsst/afw/image/imageLib.i" 

%lsst_exceptions();

SWIG_SHARED_PTR(AmpPtr, lsst::afw::cameraGeom::Amp);
SWIG_SHARED_PTR(ElectronicParamsPtr, lsst::afw::cameraGeom::ElectronicParams);
SWIG_SHARED_PTR(DetectorPtr, lsst::afw::cameraGeom::Detector);
SWIG_SHARED_PTR(DetectorLayoutPtr, lsst::afw::cameraGeom::DetectorLayout);
SWIG_SHARED_PTR_DERIVED(DetectorMosaicPtr, lsst::afw::cameraGeom::Detector,
                        lsst::afw::cameraGeom::DetectorMosaic);
SWIG_SHARED_PTR_DERIVED(CcdPtr, lsst::afw::cameraGeom::Detector, lsst::afw::cameraGeom::Ccd);
SWIG_SHARED_PTR_DERIVED(RaftPtr, lsst::afw::cameraGeom::Detector, lsst::afw::cameraGeom::Raft);

%template(AmpSet) std::vector<boost::shared_ptr<lsst::afw::cameraGeom::Amp> >;
%template(DetectorSet) std::vector<boost::shared_ptr<lsst::afw::cameraGeom::DetectorLayout> >;

%include "lsst/afw/cameraGeom/Id.h"
%include "lsst/afw/cameraGeom/Amp.h"
%include "lsst/afw/cameraGeom/Detector.h"
%include "lsst/afw/cameraGeom/DetectorMosaic.h"
%include "lsst/afw/cameraGeom/Ccd.h"
%include "lsst/afw/cameraGeom/Raft.h"

%inline %{
    lsst::afw::cameraGeom::Ccd *
    cast_Ccd(lsst::afw::cameraGeom::Detector *detector) {
        return dynamic_cast<lsst::afw::cameraGeom::Ccd *>(detector);
    }

    lsst::afw::cameraGeom::Raft *
    cast_Raft(lsst::afw::cameraGeom::Detector *detector) {
        return dynamic_cast<lsst::afw::cameraGeom::Raft *>(detector);
    }
%}
//
// We'd like to just say
//  def __iter__(self):
//      return next()
// but this crashes, at least with swig 1.3.36
//
%define %definePythonIterator(TYPE...)
%extend TYPE {
    %pythoncode {
        def __iter__(self):
            ptr = self.begin()
            end = self.end()
            while True:
                if ptr == end:
                    raise StopIteration

                yield ptr.value()
                ptr.incr()
    }
}
%enddef

%definePythonIterator(lsst::afw::cameraGeom::Ccd);
%definePythonIterator(lsst::afw::cameraGeom::DetectorMosaic);
