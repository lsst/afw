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

%define %CoordinateBase_PREINCLUDE(T, N, CLASS...)
 //%ignore lsst::afw::geom::CoordinateBase<CLASS,T,N>::asVector() const;
%enddef

%define %CoordinateBase_PREINCLUDE_2(T, CLASS...)
%CoordinateBase_PREINCLUDE(T,2,CLASS);
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,2>::asPairXY() const;
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,2>::asTupleXY() const;
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,2>::makeXY(T const[2]);
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,2>::makeXY(std::pair<T,T> const &);
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,2>::makeXY(boost::tuple<T,T> const &);
%enddef

%define %CoordinateBase_PREINCLUDE_3(T, CLASS...)
%CoordinateBase_PREINCLUDE(T,3,CLASS);
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,3>::asTupleXYZ() const;
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,3>::makeXYZ(T const[3]);
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,3>::makeXYZ(boost::tuple<T,T,T> const &);
%enddef

%define %CoordinateBase_POSTINCLUDE(T, N, NAME, CLASS...)
%template(NAME ## Base) lsst::afw::geom::CoordinateBase<CLASS,T,N>;
%template(NAME) CLASS;
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
    }
}
%enddef

%define %CoordinateBase_POSTINCLUDE_2(T, NAME, CLASS...)
%CoordinateBase_POSTINCLUDE(T,2,NAME,CLASS);
%extend CLASS {
    %pythoncode {
    def __repr__(self):
        return "NAME(x=%0.10g, y=%0.10g)" % tuple(self)
    def __str__(self):
        return "(x=%g, y=%g)" % tuple(self)
    }
}
%enddef

%define %CoordinateBase_POSTINCLUDE_3(T, NAME, CLASS...)
%CoordinateBase_POSTINCLUDE(T,3,NAME,CLASS);
%extend CLASS {
    %pythoncode {
    def __repr__(self):
        return "NAME(x=%0.10g, y=%0.10g, z=%0.10g)" % tuple(self)
    def __str__(self):
        return "(x=%g, y=%g, z=%g)" % tuple(self)
    }
}
%enddef
