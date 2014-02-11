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
#include "lsst/afw/geom/LinearTransform.h"
%}

%declareNumPyConverters(lsst::afw::geom::LinearTransform::ParameterVector);
%declareNumPyConverters(lsst::afw::geom::LinearTransform::Matrix);
%declareNumPyConverters(Eigen::Vector2d);
%declareNumPyConverters(Eigen::Matrix2d);
%declareNumPyConverters(Eigen::Matrix3d);

%shared_ptr(lsst::afw::geom::LinearTransform);

%rename(__mul__) lsst::afw::geom::LinearTransform::operator*;
%ignore lsst::afw::geom::LinearTransform::operator[];
%ignore lsst::afw::geom::LinearTransform::dTransform;
%ignore lsst::afw::geom::LinearTransform::operator=;

%copyctor lsst::afw::geom::LinearTransform;
%newobject lsst::afw::geom::LinearTransform::makeRotation;
%newobject lsst::afw::geom::LinearTransform::makeScaling;
%newobject lsst::afw::geom::LinearTransform::invert;

%include "lsst/afw/geom/Angle.i"
%include "lsst/afw/geom/LinearTransform.h"

%extend lsst::afw::geom::LinearTransform {    
    void set(double xx, double yx, double xy, double yy) {
        (*self)[lsst::afw::geom::LinearTransform::XX] = xx;
        (*self)[lsst::afw::geom::LinearTransform::XY] = xy;        
        (*self)[lsst::afw::geom::LinearTransform::YX] = yx; 
        (*self)[lsst::afw::geom::LinearTransform::YY] = yy;       
    }
    
    %feature("shadow") _setitem_nochecking %{
        def __setitem__(self, k, v):
            if k < 0 or k > 3: raise IndexError
            $action(self, k, v)
    %} 
    void _setitem_nochecking(int i, double value) {
        self->operator[](i) = value;
    }

        
    %feature("shadow") _getitem_nochecking %{
        def __getitem__(self, k):
            try:
                i,j = k
                if i < 0 or i > 1: raise IndexError
                if j < 0 or j > 1: raise IndexError
                return $action(self, i,j)
            except TypeError:
                if k < 0 or k > 3: raise IndexError
                return $action(self, k)

    %}
    double _getitem_nochecking(int row, int col) {
        return (self->getMatrix())(row, col);
    }
    double _getitem_nochecking(int i) {
        return self->operator[](i);
    }   
         
    %pythoncode {
        def __str__(self):
            return str(self.getMatrix())
        def __reduce__(self):
            return (LinearTransform, (self.getMatrix(),))
        def __repr__(self):
            return "LinearTransform(\n%r\n)" % (self.getMatrix(),)
    }
}
