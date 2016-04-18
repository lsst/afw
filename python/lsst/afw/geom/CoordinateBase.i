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
#include "lsst/afw/geom/CoordinateBase.h"
%}

%define %CoordinateBase_POSTINCLUDE(T, N, CLASS...)
%extend CLASS {

    %feature("shadow") _getitem_nochecking %{
        def __getitem__(self, k):
            if k < 0 or k >= N: raise IndexError(k)
            return $action(self, k)
    %}

    T _getitem_nochecking(int i) {
        return (*self)[i];
    }

    %feature("shadow") _setitem_nochecking %{
        def __setitem__(self, k, v):
            if k < 0 or k >= N: raise IndexError(k)
            $action(self, k, v)
    %}

    void _setitem_nochecking(int i, T value) {
        (*self)[i] = value;
    }

    CLASS clone() {
        return CLASS(static_cast<CLASS const &>(*self));
    }

    %pythoncode %{

    __swig_getmethods__["x"] = getX
    __swig_setmethods__["x"] = setX
    __swig_getmethods__["y"] = getY
    __swig_setmethods__["y"] = setY

    def __len__(self):
        return N

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, ", ".join("%0.10g" % v for v in self))

    def __str__(self):
        return "(%s)" % (", ".join("%0.5g" % v for v in self),)

    %}
}
%enddef
