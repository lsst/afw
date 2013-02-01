// -*- lsst-c++ -*-

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

%define imageLib_DOCSTRING
"
Basic routines to talk to lsst::afw::image classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.image", docstring=imageLib_DOCSTRING) imageLib

%include "lsst/p_lsstSwig.i"
%include "lsst/daf/base/persistenceMacros.i"
%include "lsst/afw/image/LsstImageTypes.h"

%lsst_exceptions();

namespace lsst { namespace afw { namespace image {

class Filter;
class FilterProperty;
class Wcs;
class TanWcs;
class Color;
class Calib;
class Defect;
class ExposureInfo;

}}} // namespace lsst::afw::image

%shared_ptr(lsst::afw::image::Wcs);
%shared_ptr(lsst::afw::image::TanWcs);
%shared_ptr(lsst::afw::image::Calib);
%shared_ptr(lsst::afw::image::DefectBase);
