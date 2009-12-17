
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
    
    void set(int i, double value) {
        self->operator[](i) = value;
    }
        
    double get(int row, int col) {
        return (self->matrix())(row, col);
    }
    double get(int i) {
        return self->operator[](i);
    }   
         
    %pythoncode {
    def __getitem__(self, k):
        return self.get(k)
             
    def __setitem__(self, k, v):
        self.set(k, v)    
    }
}
