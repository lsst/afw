
%{
#include "lsst/afw/geom/AffineTransform.h"
%}

%rename(__mul__) lsst::afw::geom::AffineTransform::operator*;
%ignore lsst::afw::geom::AffineTransform::operator[];
%ignore lsst::afw::geom::AffineTransform::dTransform;
%ignore lsst::afw::geom::AffineTransform::getVector;
%ignore lsst::afw::geom::AffineTransform::setVector;
%ignore lsst::afw::geom::AffineTransform::matrix;
%ignore lsst::afw::geom::AffineTransform::operator=;

%copyctor lsst::afw::geom::AffineTransform;
%newobject lsst::afw::geom::AffineTransform::makeRotation;
%newobject lsst::afw::geom::AffineTransform::makeScaling;
%newobject lsst::afw::geom::AffineTransform::invert;

%include "lsst/afw/geom/AffineTransform.h"

%extend lsst::afw::geom::AffineTransform {    
    void set(double xx, double yx, double xy, double yy, double x, double y) {
        (self->matrix())(0, 0) = xx;
        (self->matrix())(0, 1) = xy;
        (self->matrix())(0, 2) = x;
        (self->matrix())(1, 0) = yx; 
        (self->matrix())(1, 1) = yy;
        (self->matrix())(1, 2) = y;
    }
    
    void _setitem_nochecking(int i, double value) {
        self->operator[](i) = value;
    }
        
    double _getitem_nochecking(int row, int col) {
        return (self->matrix())(row, col);
    }
    double _getitem_nochecking(int i) {
        return self->operator[](i);
    }   
         
    %pythoncode {
    def __getitem__(self, k):
        try:
            i,j = k
            if i < 0 or i > 2: raise IndexError
            if j < 0 or j > 2: raise IndexError
            return self._getitem_nochecking(i,j)
        except TypeError:
            if k < 0 or k > 5: raise IndexError
            return self._getitem_nochecking(k)
             
    def __setitem__(self, k, v):
        if k < 0 or k > 5: raise IndexError
        self._setitem_nochecking(k, v)
    def matrix(self):
        return ((self[0,0], self[0,1], self[0,2]),
                (self[1,0], self[1,1], self[1,2]),
                (0,0,1))
    def __str__(self):
        return str(self.matrix())
    def __repr__(self):
        return "AffineTransform(%r)" % (self.matrix(),)
    }
}
