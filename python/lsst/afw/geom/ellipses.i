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
 

%ignore lsst::afw::geom::ellipses::BaseEllipse::operator[];
%ignore lsst::afw::geom::ellipses::BaseEllipse::getVector;
%ignore lsst::afw::geom::ellipses::BaseEllipse::setVector;
%ignore lsst::afw::geom::ellipses::BaseEllipse::getCore;
%ignore lsst::afw::geom::ellipses::BaseEllipse::getCenter();
%ignore lsst::afw::geom::ellipses::BaseEllipse::transform;
%ignore lsst::afw::geom::ellipses::BaseCore::operator[];
%ignore lsst::afw::geom::ellipses::BaseCore::getVector;
%ignore lsst::afw::geom::ellipses::BaseCore::setVector;
%ignore lsst::afw::geom::ellipses::BaseCore::transform;
//%ignore lsst::afw::geom::ellipses::BaseCore::dAssign;

%declareEigenMatrix(lsst::afw::geom::ellipses::BaseCore::Jacobian)

%rename(set) lsst::afw::geom::ellipses::BaseEllipse::operator=;
%rename(set) lsst::afw::geom::ellipses::BaseCore::operator=;

%include "boost_shared_ptr.i"
%include "lsst/afw/eigen.i"

SWIG_SHARED_PTR(BaseEllipsePtr, lsst::afw::geom::ellipses::BaseEllipse);
SWIG_SHARED_PTR(BaseCorePtr, lsst::afw::geom::ellipses::BaseCore);

%include "lsst/afw/geom/ellipses/BaseEllipse.h"

%extend lsst::afw::geom::ellipses::BaseCore {
    double _getitem_nochecking(int i) { return self->operator[](i); }
    void _setitem_nochecking(int i, double value) {
        self->operator[](i) = value;
    }
    %pythoncode {
    def __len__(self):
        return 3
    def __getitem__(self, k):
        if k < 0 or k > 2: raise IndexError(k)
        return self._getitem_nochecking(k)
    def __setitem__(self, k, v):
        if k < 0 or k > 2: raise IndexError(k)
        self._setitem_nochecking(k, v)
    }
}

%extend lsst::afw::geom::ellipses::BaseEllipse {
    double _getitem_nochecking(int i) { return self->operator[](i); }
    void _setitem_nochecking(int i, double value) {
        self->operator[](i) = value;
    }
    %pythoncode {
    def __len__(self):
        return 5
    def __getitem__(self, k):
        if k < 0 or k > 4: raise IndexError(k)
        return self._getitem_nochecking(k)
    def __setitem__(self, k, v):
        if k < 0 or k > 4: raise IndexError(k)
        self._setitem_nochecking(k, v)
    }
}

%define %Ellipse_PREINCLUDE(NAME)
%feature(notabstract) lsst::afw::geom::ellipses::NAME;
SWIG_SHARED_PTR_DERIVED(
    NAME ## EllipsePtr,
    lsst::afw::geom::ellipses::BaseEllipse,
    lsst::afw::geom::ellipses::NAME ## Ellipse
)
SWIG_SHARED_PTR_DERIVED(
    NAME ## Ptr,
    lsst::afw::geom::ellipses::BaseCore,
    lsst::afw::geom::ellipses::NAME 
)
%ignore lsst::afw::geom::ellipses::NAME ## Ellipse::getCore;
%rename(set) lsst::afw::geom::ellipses::NAME ## Ellipse::operator=;
%rename(set) lsst::afw::geom::ellipses::NAME::operator=;
%enddef

%define %Ellipse_POSTINCLUDE(NAME)
%extend lsst::afw::geom::ellipses::NAME {
    lsst::afw::geom::ellipses::NAME::Ptr _transform(LinearTransform const & t) {
        return boost::static_pointer_cast<lsst::afw::geom::ellipses::NAME>(self->transform(t).copy());
    }
    lsst::afw::geom::ellipses::NAME::Ptr _transform(AffineTransform const & t) {
        return boost::static_pointer_cast<lsst::afw::geom::ellipses::NAME>(self->transform(t).copy());
    }
    %pythoncode {
    def transform(self,t): return self._transform(t)
    def __repr__(self):
        return "NAME(%.10g,%.10g,%.10g)" % tuple(self)
    def __str__(self):
        return "NAME(%g,%g,%g)" % tuple(self)
    }
}
%extend lsst::afw::geom::ellipses::NAME ## Ellipse {
    lsst::afw::geom::ellipses::NAME ## Ellipse::Ptr _transform(AffineTransform const & t) {
        return boost::static_pointer_cast<lsst::afw::geom::ellipses::NAME ## Ellipse>(
            self->transform(t).copy()
        );
    }
    lsst::afw::geom::ellipses::NAME::Ptr _getCore() {
        return self->getCore().clone();
    }
    %pythoncode {
    def transform(self,t): return self._transform(t)
    def getCore(self): return self._getCore()
    def __repr__(self):
        return "NAME" "Ellipse(%r,%r)" % (self.getCore(), self.getCenter())
    def __str__(self):
        return "(%s,%s)" % (self.getCore(), self.getCenter())
    }
}
%enddef
 
%Ellipse_PREINCLUDE(Axes);
%Ellipse_PREINCLUDE(Distortion);
%Ellipse_PREINCLUDE(LogShear);
%Ellipse_PREINCLUDE(Quadrupole);

%include "lsst/afw/geom/ellipses/Axes.h"
%include "lsst/afw/geom/ellipses/Distortion.h"
%include "lsst/afw/geom/ellipses/Quadrupole.h"
%include "lsst/afw/geom/ellipses/LogShear.h"

%Ellipse_POSTINCLUDE(Axes);
%Ellipse_POSTINCLUDE(Distortion);
%Ellipse_POSTINCLUDE(LogShear);
%Ellipse_POSTINCLUDE(Quadrupole);
