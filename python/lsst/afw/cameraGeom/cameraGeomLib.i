// -*- lsst-c++ -*-
%define cameraGeomLib_DOCSTRING
"
Python bindings for classes describing the the geometry of a mosaic camera
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.cameraGeom", docstring=cameraGeomLib_DOCSTRING) cameraGeomLib

%{
#include "lsst/afw/cameraGeom.h"
%}

%include "lsst/p_lsstSwig.i"
%import  "lsst/afw/image/imageLib.i" 

%lsst_exceptions();

SWIG_SHARED_PTR(DetectorPtr, lsst::afw::cameraGeom::Detector);

SWIG_SHARED_PTR_DERIVED(AmpPtr, lsst::afw::cameraGeom::Detector, lsst::afw::cameraGeom::Amp);
SWIG_SHARED_PTR(ElectronicParamsPtr, lsst::afw::cameraGeom::ElectronicParams);
SWIG_SHARED_PTR_DERIVED(DetectorMosaicPtr, lsst::afw::cameraGeom::Detector,
                        lsst::afw::cameraGeom::DetectorMosaic);
SWIG_SHARED_PTR_DERIVED(CcdPtr, lsst::afw::cameraGeom::Detector, lsst::afw::cameraGeom::Ccd);
SWIG_SHARED_PTR_DERIVED(RaftPtr, lsst::afw::cameraGeom::Detector, lsst::afw::cameraGeom::Raft);
SWIG_SHARED_PTR_DERIVED(CameraPtr, lsst::afw::cameraGeom::Detector, lsst::afw::cameraGeom::Camera);

%template(AmpSet) std::vector<boost::shared_ptr<lsst::afw::cameraGeom::Amp> >;
%template(DetectorSet) std::vector<boost::shared_ptr<lsst::afw::cameraGeom::Detector> >;

%include "lsst/afw/cameraGeom/Id.h"

%extend lsst::afw::cameraGeom::Id {
    %pythoncode {
        def __str__(self):
            return "%d, %s" % (self.getSerial(), self.getName())

        def __repr__(self):
            return "Id(%s)" % (str(self))
    }
}

%include "lsst/afw/cameraGeom/Orientation.h"
%include "lsst/afw/cameraGeom/Detector.h"
%include "lsst/afw/cameraGeom/Amp.h"
%include "lsst/afw/cameraGeom/DetectorMosaic.h"
%include "lsst/afw/cameraGeom/Ccd.h"
%include "lsst/afw/cameraGeom/Raft.h"
%include "lsst/afw/cameraGeom/Camera.h"

%inline %{
    lsst::afw::cameraGeom::Ccd::Ptr
    cast_Ccd(lsst::afw::cameraGeom::Detector::Ptr detector) {
        return boost::shared_dynamic_cast<lsst::afw::cameraGeom::Ccd>(detector);
    }

    lsst::afw::cameraGeom::Raft::Ptr
    cast_Raft(lsst::afw::cameraGeom::Detector::Ptr detector) {
        return boost::shared_dynamic_cast<lsst::afw::cameraGeom::Raft>(detector);
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

        def __getitem__(self, i):
            return [e for e in self][i]
    }
}
%enddef

%definePythonIterator(lsst::afw::cameraGeom::Ccd);
%definePythonIterator(lsst::afw::cameraGeom::DetectorMosaic);

%pythoncode {
class ReadoutCorner(object):
    """A python object corresponding to Amp::ReadoutCorner"""
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return ["LLC", "LRC", "URC", "ULC"][self.value]
}
