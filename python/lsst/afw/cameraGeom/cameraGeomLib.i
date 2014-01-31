// -*- lsst-++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
// used by lsst/afw/image/imageLib.i to avoid circular import
// can be removed if and when this file does not need imageLib.i
#define CAMERA_GEOM_LIB_I

%define cameraGeomLib_DOCSTRING
"
Python bindings for classes describing the the geometry of a mosaic camera
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.cameraGeom", docstring=cameraGeomLib_DOCSTRING) cameraGeomLib

%{
#include <utility>
#include <vector>
#include <map>
#include "boost/shared_ptr.hpp"
#include "lsst/pex/logging.h"
#include "lsst/afw/geom/TransformRegistry.h"
#include "lsst/afw/cameraGeom.h"
%}

%include "lsst/p_lsstSwig.i"
%include "lsst/afw/utils.i" 
%include "std_map.i"

%import "lsst/afw/geom/geomLib.i"

%lsst_exceptions();

%shared_ptr(lsst::afw::cameraGeom::Detector);
%shared_ptr(lsst::afw::cameraGeom::RawAmplifier);
%shared_ptr(lsst::afw::cameraGeom::Amplifier);

%template(CameraSysList) std::vector<lsst::afw::cameraGeom::CameraSys>;
%template(CameraTransformMap)
    std::map<lsst::afw::cameraGeom::CameraSys, CONST_PTR(lsst::afw::geom::XYTransform)>;
%template(CameraTransformRegistry) lsst::afw::geom::TransformRegistry<lsst::afw::cameraGeom::CameraSys>;
%template(AmplifierList) std::vector<CONST_PTR(lsst::afw::cameraGeom::Amplifier)>;
%template(DetectorList) std::vector<CONST_PTR(lsst::afw::cameraGeom::Detector)>;

%rename(__getitem__) lsst::afw::cameraGeom::Detector::operator[];
// the following rename silently fails (and so does %ignore) so use %extend to add __len__=size
// %rename(__len__) lsst::afw::cameraGeom::Detector::size();

%include "lsst/afw/cameraGeom/CameraSys.h"
%include "lsst/afw/cameraGeom/CameraPoint.h"
%include "lsst/afw/cameraGeom/Orientation.h"
%include "lsst/afw/cameraGeom/RawAmplifier.h"
%include "lsst/afw/cameraGeom/Amplifier.h"
%include "lsst/afw/cameraGeom/Detector.h"

%extend lsst::afw::cameraGeom::CameraSysPrefix {
    std::string __repr__() {
        std::ostringstream os;
        os << *$self;
        return os.str();
    }
}

%extend lsst::afw::cameraGeom::CameraSys {
    std::string __repr__() const {
        std::ostringstream os;
        os << *$self;
        return os.str();
    }

    %pythoncode {
        def __hash__(self):
            return hash(repr(self))
    }
}

%extend lsst::afw::cameraGeom::Detector {
    size_t __len__() const { return $self->size(); }

    %pythoncode {
        def __iter__(self):
            for i in xrange(len(self)):
                yield self[i]
    }
}
