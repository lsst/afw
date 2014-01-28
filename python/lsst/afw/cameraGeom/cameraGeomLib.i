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
#include "boost/shared_ptr.hpp"
#include "lsst/pex/logging.h"
#include "lsst/afw/geom/TransformRegistry.h"
#include "lsst/afw/cameraGeom.h"
%}

%include "lsst/p_lsstSwig.i"
%include "lsst/afw/utils.i" 

%import "lsst/afw/geom/geomLib.i"
%include "lsst/afw/geom/TransformRegistry.h"

%lsst_exceptions();

%pythoncode {
# See comment in Orientation.h
import lsst.afw.geom            # needed for initialising Orientation
radians = lsst.afw.geom.radians
}

%shared_ptr(lsst::afw::cameraGeom::Detector);
%shared_ptr(lsst::afw::cameraGeom::RawAmplifier);
%shared_ptr(lsst::afw::cameraGeom::Amplifier);

%include "lsst/afw/cameraGeom/CameraSys.h"
%include "lsst/afw/cameraGeom/CameraPoint.h"
%include "lsst/afw/cameraGeom/Orientation.h"
%include "lsst/afw/cameraGeom/RawAmplifier.h"
%include "lsst/afw/cameraGeom/Amplifier.h"
%include "lsst/afw/cameraGeom/Detector.h"

%template(CameraTransformList) std::vector<std::pair<lsst::afw::cameraGeom::CameraSys, boost::shared_ptr<lsst::afw::geom::XYTransform> > >;
%template(CameraTransformRegistry) lsst::afw::geom::TransformRegistry<lsst::afw::cameraGeom::CameraSys>;
%template(AmplifierList) std::vector<boost::shared_ptr<lsst::afw::cameraGeom::Amplifier> >;
%template(DetectorList) std::vector<boost::shared_ptr<lsst::afw::cameraGeom::Detector> >;

