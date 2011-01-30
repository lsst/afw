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
    T _getitem_nochecking(int i) {
        return (*self)[i];
    }
    void _setitem_nochecking(int i, T value) {
        (*self)[i] = value;
    }
    CLASS clone() {
        return CLASS(static_cast<CLASS const &>(*self));
    }
    %pythoncode {
    def __len__(self):
        return N;
    def __getitem__(self,i):
        if i < 0 or i >= N: raise IndexError(i)
        return self._getitem_nochecking(i)
    def __setitem__(self,i,value):
        if i < 0 or i >= N: raise IndexError(i)
        self._setitem_nochecking(i,value)
    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, ", ".join("%0.10g" % v for v in self))
    def __str__(self):
        return "(%s)" % (", ".join("%0.5g" % v for v in self),)
    }
}
%enddef
