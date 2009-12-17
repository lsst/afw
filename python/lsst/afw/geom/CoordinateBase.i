
%{
#include "lsst/afw/geom/CoordinateBase.h"
%}

%define %CoordinateBase_PREINCLUDE(T, N, CLASS...)
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,N>::asVector();
%enddef

%define %CoordinateBase_PREINCLUDE_2(T, CLASS...)
%CoordinateBase_PREINCLUDE(T,2,CLASS);
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,2>::asPairXY();
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,2>::asTupleXY();
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,2>::makeXY(T const[2]);
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,2>::makeXY(std::pair<T,T> const &);
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,2>::makeXY(boost::tuple<T,T> const &);
%enddef

%define %CoordinateBase_PREINCLUDE_3(T, CLASS...)
%CoordinateBase_PREINCLUDE(T,3,CLASS);
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,3>::asTupleXYZ();
%ignore lsst::afw::geom::CoordinateBase<CLASS,T,3>::makeXYZ;
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
    def asTupleXY(self):
        return tuple(self)
    def __repr__(self):
        return "NAME(x=%0.10g, y=%0.10g)" % self.asTupleXY()
    def __str__(self):
        return "(x=%g, y=%g)" % self.asTupleXY()
    }
}
%enddef

%define %CoordinateBase_POSTINCLUDE_3(T, NAME, CLASS...)
%CoordinateBase_POSTINCLUDE(T,3,NAME,CLASS);
%extend CLASS {
    %pythoncode {
    def asTupleXYZ():
        return tuple(self)
    def __repr__(self):
        return "NAME(x=%0.10g, y=%0.10g, z=%0.10g)" % self.asTupleXYZ()
    def __str__(self):
        return "(x=%g, y=%g, z=%g)" % self.asTupleXYZ()
    }
}
%enddef
