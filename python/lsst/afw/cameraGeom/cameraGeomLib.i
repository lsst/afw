// -*- lsst-c++ -*-
%define cameraGeomLib_DOCSTRING
"
Python bindings for classes describing the the geometry of a mosaic camera
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw", docstring=cameraGeomLib_DOCSTRING) cameraGeomLib

// Suppress swig complaints; see afw/image/imageLib.i for more 
//#pragma SWIG nowarn=314                 // print is a python keyword (--> _print)
//#pragma SWIG nowarn=362                 // operator=  ignored 

%{
#include "lsst/afw/cameraGeom.h"
%}

%include "lsst/p_lsstSwig.i"
%import  "lsst/afw/image/imageLib.i" 

%lsst_exceptions();

%std_nodefconst_type(lsst::afw::cameraGeom::Amp);
%template(AmpSet) std::vector<lsst::afw::cameraGeom::Amp>;

%std_nodefconst_type(lsst::afw::cameraGeom::Detector);
%template(DetectorSet) std::vector<lsst::afw::cameraGeom::Detector>;

%include "lsst/afw/cameraGeom/Id.h"
%include "lsst/afw/cameraGeom/Amp.h"
%include "lsst/afw/cameraGeom/Detector.h"
%include "lsst/afw/cameraGeom/Ccd.h"
%include "lsst/afw/cameraGeom/Raft.h"

%extend lsst::afw::cameraGeom::Ccd {
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

%extend lsst::afw::cameraGeom::Raft {
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
