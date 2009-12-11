
%{
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "lsst/afw/math/AffineTransform.h"
%}

%rename(__mul__) lsst::afw::math::AffineTransform::operator*;
%ignore lsst::afw::math::AffineTransform::operator[];
%ignore lsst::afw::math::AffineTransform::d;
%ignore lsst::afw::math::AffineTransform::getVector;
%ignore lsst::afw::math::AffineTransform::matrix;

%copyctor lsst::afw::math::AffineTransform;
%newobject lsst::afw::math::AffineTransform::makeRotation;
%newobject lsst::afw::math::AffineTransform::makeScaling;
%newobject lsst::afw::math::AffineTransform::invert;

%include "lsst/afw/math/AffineTransform.h"

%extend lsst::afw::math::AffineTransform {    
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
