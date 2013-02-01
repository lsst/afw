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
 
%define detectionLib_DOCSTRING
"
Python interface to lsst::afw::detection classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.detection", docstring=detectionLib_DOCSTRING) detectionLib

%include "lsst/p_lsstSwig.i"
%include "lsst/daf/base/persistenceMacros.i"

%lsst_exceptions()

namespace lsst { namespace afw { namespace detection {

class Peak;
class Footprint;
class Psf;
class KernelPsf;
class DoubleGaussianPsf;
class FootprintSet;

}}} // namespace lsst::afw::detection

%shared_ptr(lsst::afw::detection::Peak);
%shared_ptr(lsst::afw::detection::Footprint);
%shared_ptr(lsst::afw::detection::Psf);
%shared_ptr(lsst::afw::detection::KernelPsf);
%shared_ptr(lsst::afw::detection::DoubleGaussianPsf);
%shared_ptr(lsst::afw::detection::FootprintSet);
