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

%{
#include "lsst/afw/geom/ellipses/BaseCore.h"
#include "lsst/afw/geom/ellipses/Ellipse.h"
%}

%feature("notabstract") lsst::afw::geom::ellipses::BaseCore;

%ignore lsst::afw::geom::ellipses::BaseCore::transform;
%ignore lsst::afw::geom::ellipses::BaseCore::convolve;
%ignore lsst::afw::geom::ellipses::BaseCore::getGridTransform;
%ignore lsst::afw::geom::ellipses::BaseCore::readParameters;
%ignore lsst::afw::geom::ellipses::BaseCore::writeParameters;
%rename(assign) lsst::afw::geom::ellipses::BaseCore::operator=;

SWIG_SHARED_PTR(BaseCorePtr, lsst::afw::geom::ellipses::BaseCore);

%include "lsst/afw/geom/ellipses/BaseCore.h"

// %ignore lsst::afw::geom::ellipses::Ellipse::make;
// %ignore lsst::afw::geom::ellipses::Ellipse::getCore;
// %ignore lsst::afw::geom::ellipses::Ellipse::transform;
// %ignore lsst::afw::geom::ellipses::Ellipse::getGridTransform;
// %ignore lsst::afw::geom::ellipses::Ellipse::readParameters;
// %ignore lsst::afw::geom::ellipses::Ellipse::writeParameters;

// %rename(assign) lsst::afw::geom::ellipses::Ellipse::operator=;
// %rename(_getCorePtr) lsst::afw::geom::ellipses::Ellipse::getCorePtr;

// SWIG_SHARED_PTR(EllipsePtr, lsst::afw::geom::ellipses::Ellipse);

//%include "lsst/afw/geom/ellipses/Ellipse.h"

// %extend lsst::afw::geom::ellipses::Ellipse {
//     lsst::afw::geom::ellipses::Ellipse _transform(lsst::afw::geom::AffineTransform const & t) {
//         return self->transform(t);
//     }
//     void _transformInPlace(lsst::afw::geom::AffineTransform const & t) {
//         self->transform(t).inPlace();
//     }
//     lsst::afw::geom::AffineTransform _getGridTransform() {
//         return self->getGridTransform();
//     }
//     %pythoncode {
//     def transform(self, t): return self._transform(t)
//     def transformInPlace(self, t): self._transformInPlace(t)
//     def getGridTransform(self, t): return self._getGridTransform(t)
//     def getCore(self): return self._getCorePtr()
//     def __repr__(self):
//         return "Ellipse(%r, %r)" % (self.getCore(), self.getCenter())
//     def __str__(self):
//         return "(%s, %s)" % (self.getCore(), self.getCenter())
//     }
// }

// %declareNumPyConverters(lsst::afw::geom::ellipses::BaseCore::Jacobian)
// %declareNumPyConverters(lsst::afw::geom::ellipses::Quadrupole::Matrix);

// %define %EllipseCore_PREINCLUDE(NAME)
// %feature(notabstract) lsst::afw::geom::ellipses::NAME;
// SWIG_SHARED_PTR_DERIVED(
//     NAME ## Ptr,
//     lsst::afw::geom::ellipses::BaseCore,
//     lsst::afw::geom::ellipses::NAME 
// )
// %rename(assign) lsst::afw::geom::ellipses::NAME::operator=;
// %enddef

// %define %EllipseCore_POSTINCLUDE(NAME)
// %extend lsst::afw::geom::ellipses::NAME {
//     lsst::afw::geom::ellipses::NAME::Ptr _transform(lsst::afw::geom::LinearTransform const & t) {
//         return boost::static_pointer_cast<lsst::afw::geom::ellipses::NAME>(self->transform(t).copy());
//     }
//     void _transformInPlace(lsst::afw::geom::LinearTransform const & t) {
//         self->transform(t).inPlace();
//     }
//     lsst::afw::geom::ellipses::NAME::Ptr _convolve(lsst::afw::geom::ellipses::BaseCore const & other) {
//         return boost::static_pointer_cast<lsst::afw::geom::ellipses::NAME>(self->convolve(other).copy());
//     }
//     static lsst::afw::geom::ellipses::NAME::Ptr cast(lsst::afw::geom::ellipses::BaseCore const & p) {
//         return boost::dynamic_pointer_cast<lsst::afw::geom::ellipses::NAME>(p);
//     }
//     %pythoncode {
//     def transform(self, t): return self._transform(t)
//     def transformInPlace(self, t): self._transformInPlace(t)
//     def convolve(self, t): return self._convolve(t)
//     }
// }
// %enddef
 
 //%Ellipse_PREINCLUDE(Axes);
 //%Ellipse_PREINCLUDE(Quadrupole);

 //%include "lsst/afw/geom/ellipses/Axes.h"
 //%include "lsst/afw/geom/ellipses/Quadrupole.h"

 //%Ellipse_POSTINCLUDE(Axes);
 //%Ellipse_POSTINCLUDE(Quadrupole);

//%include "lsst/afw/geom/ellipses/EllipticityBase.h"
//%include "lsst/afw/geom/ellipses/Distortion.h"
//%include "lsst/afw/geom/ellipses/LogShear.h"
