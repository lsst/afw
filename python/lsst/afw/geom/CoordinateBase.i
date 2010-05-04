
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
