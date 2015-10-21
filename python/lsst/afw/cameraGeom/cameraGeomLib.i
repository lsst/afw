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
#include "lsst/afw/geom.h"
#include "lsst/pex/logging.h"
#include "lsst/afw/geom/TransformMap.h"
#include "lsst/afw/cameraGeom.h"
%}

%include "lsst/p_lsstSwig.i"
%include "lsst/afw/utils.i"
%include "std_map.i"

%import "lsst/afw/geom/geomLib.i"
%import "lsst/afw/table/AmpInfo.i"

%initializeNumPy(afw_cameraGeom)
%lsst_exceptions();

%shared_ptr(lsst::afw::cameraGeom::Detector);

%template(CameraSysList) std::vector<lsst::afw::cameraGeom::CameraSys>;
// I would prefer to use lsst::afw::geom::TransformMap<lsst::afw::cameraGeom::CameraSys>::Transforms
// in the following, but SIG complains of a syntax error
%template(CameraTransforms)
    std::map<lsst::afw::cameraGeom::CameraSys, CONST_PTR(lsst::afw::geom::XYTransform)>;
%template(CameraTransformMap) lsst::afw::geom::TransformMap<lsst::afw::cameraGeom::CameraSys>;
%template(DetectorList) std::vector<CONST_PTR(lsst::afw::cameraGeom::Detector)>;

%rename(__getitem__) lsst::afw::cameraGeom::Detector::operator[];
%rename(__len__) lsst::afw::cameraGeom::Detector::size;
%ignore lsst::afw::cameraGeom::Detector::begin;
%ignore lsst::afw::cameraGeom::Detector::end;

%include "lsst/afw/cameraGeom/CameraSys.h"
%include "lsst/afw/cameraGeom/CameraPoint.h"
%include "lsst/afw/cameraGeom/Orientation.h"
%include "lsst/afw/cameraGeom/Detector.h"

// macros from p_lsstSWig.i
%addStreamRepr(lsst::afw::cameraGeom::CameraSysPrefix)
%useValueEquality(lsst::afw::cameraGeom::CameraSysPrefix)
%addStreamRepr(lsst::afw::cameraGeom::CameraSys)
%useValueEquality(lsst::afw::cameraGeom::CameraSys)
%addStreamRepr(lsst::afw::cameraGeom::CameraPoint)
%useValueEquality(lsst::afw::cameraGeom::CameraPoint)

%extend lsst::afw::cameraGeom::CameraSys {
    %pythoncode %{
        def __hash__(self):
            return hash(repr(self))
    %}
}

%extend lsst::afw::cameraGeom::Detector {
    %pythoncode %{
        def __iter__(self):
            for i in xrange(len(self)):
                yield self[i]
    %}
}
