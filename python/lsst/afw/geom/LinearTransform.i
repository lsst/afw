
%{
#include "lsst/afw/geom/LinearTransform.h"
%}

SWIG_SHARED_PTR(LinearTransformPtr, lsst::afw::geom::LinearTransform);

%rename(__mul__) lsst::afw::geom::LinearTransform::operator*;
%ignore lsst::afw::geom::LinearTransform::operator[];
%ignore lsst::afw::geom::LinearTransform::getMatrix;
%ignore lsst::afw::geom::LinearTransform::dTransform;
%ignore lsst::afw::geom::LinearTransform::getVector;
%ignore lsst::afw::geom::LinearTransform::setVector;
%ignore lsst::afw::geom::LinearTransform::operator=;

%copyctor lsst::afw::geom::LinearTransform;
%newobject lsst::afw::geom::LinearTransform::makeRotation;
%newobject lsst::afw::geom::LinearTransform::makeScaling;
%newobject lsst::afw::geom::LinearTransform::invert;

%include "lsst/afw/geom/LinearTransform.h"

%extend lsst::afw::geom::LinearTransform {    
    void set(double xx, double yx, double xy, double yy, double x, double y) {
        (*self)[lsst::afw::geom::LinearTransform::XX] = xx;
        (*self)[lsst::afw::geom::LinearTransform::XY] = xy;        
        (*self)[lsst::afw::geom::LinearTransform::YX] = yx; 
        (*self)[lsst::afw::geom::LinearTransform::YY] = yy;       
    }
    
    void _setitem_nochecking(int i, double value) {
        self->operator[](i) = value;
    }
        
    double _getitem_nochecking(int row, int col) {
        return (self->getMatrix())(row, col);
    }
    double _getitem_nochecking(int i) {
        return self->operator[](i);
    }   
         
    %pythoncode {
    def __getitem__(self, k):
        try:
            i,j = k
            if i < 0 or i > 1: raise IndexError
            if j < 0 or j > 1: raise IndexError
            return self._getitem_nochecking(i,j)
        except TypeError:
            if k < 0 or k > 3: raise IndexError
            return self._getitem_nochecking(k)
    def __setitem__(self, k, v):
        if k < 0 or k > 5: raise IndexError
        self._setitem_nochecking(k, v)
    def __str__(self):
        return str(self.getMatrix())
    def __repr__(self):
        return "LinearTransform(\n%r\n)" % (self.getMatrix(),)
    }
}
