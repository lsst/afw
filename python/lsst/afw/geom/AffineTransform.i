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
#include "lsst/afw/geom/AffineTransform.h"
%}

%declareNumPyConverters(lsst::afw::geom::AffineTransform::ParameterVector);
%declareNumPyConverters(lsst::afw::geom::AffineTransform::Matrix);

%shared_ptr(lsst::afw::geom::AffineTransform);

%rename(__mul__) lsst::afw::geom::AffineTransform::operator*;
%ignore lsst::afw::geom::AffineTransform::operator[];
%ignore lsst::afw::geom::AffineTransform::dTransform;
%ignore lsst::afw::geom::AffineTransform::operator=;

%copyctor lsst::afw::geom::AffineTransform;
%newobject lsst::afw::geom::AffineTransform::makeRotation;
%newobject lsst::afw::geom::AffineTransform::makeScaling;
%newobject lsst::afw::geom::AffineTransform::makeTranslation;
%newobject lsst::afw::geom::AffineTransform::invert;

%include "lsst/afw/geom/Angle.i"
%include "lsst/afw/geom/AffineTransform.h"

%extend lsst::afw::geom::AffineTransform {    
    void set(double xx, double yx, double xy, double yy, double x, double y) {
        (*self)[lsst::afw::geom::AffineTransform::XX] = xx;
        (*self)[lsst::afw::geom::AffineTransform::XY] = xy;
        (*self)[lsst::afw::geom::AffineTransform::X] = x;
        (*self)[lsst::afw::geom::AffineTransform::YX] = yx; 
        (*self)[lsst::afw::geom::AffineTransform::YY] = yy;
        (*self)[lsst::afw::geom::AffineTransform::Y] = y;
    }
    
    %feature("shadow") _setitem_nochecking %{
        def __setitem__(self, k, v):
            if k < 0 or k > 5: raise IndexError
            $action(self, k, v)
    %}
    void _setitem_nochecking(int i, double value) {
        self->operator[](i) = value;
    }
    
    %feature("shadow") _getitem_nochecking %{
        def __getitem__(self, k):
            try:
                i,j = k
                if i < 0 or i > 2: raise IndexError
                if j < 0 or j > 2: raise IndexError
                return $action(self, i,j)
            except TypeError:
                if k < 0 or k > 5: raise IndexError
                return $action(self, k)

    %}
    double _getitem_nochecking(int row, int col) {
        return (self->getMatrix())(row, col);
    }
    double _getitem_nochecking(int i) {
        return self->operator[](i);
    }   
         
    %pythoncode %{
        def __str__(self):
            return str(self.getMatrix())
        def __reduce__(self):
            return (AffineTransform, (self.getMatrix(),))
        def __repr__(self):
            return "AffineTransform(\n%r\n)" % (self.getMatrix(),)
    %}
}
