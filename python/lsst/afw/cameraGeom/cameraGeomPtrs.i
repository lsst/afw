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

%shared_ptr(lsst::afw::cameraGeom::Detector);
%shared_ptr(lsst::afw::cameraGeom::Amp);
%shared_ptr(lsst::afw::cameraGeom::ElectronicParams);
%shared_ptr(lsst::afw::cameraGeom::DetectorMosaic);
%shared_ptr(lsst::afw::cameraGeom::Ccd);
%shared_ptr(lsst::afw::cameraGeom::Raft);
%shared_ptr(lsst::afw::cameraGeom::Camera);
%shared_ptr(lsst::afw::cameraGeom::Distortion);
%shared_ptr(lsst::afw::cameraGeom::NullDistortion);
%shared_ptr(lsst::afw::cameraGeom::RadialPolyDistortion);
