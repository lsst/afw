// -*- lsst-c++ -*-
%include "affine_transform.i"

%{
#include "Eigen/Core"

#include "lsst/afw/geom/ellipses.h"
%}

%include "std_complex.i"

//define mappings from Eigen::Vectors to tuples
%typemap(out)
     lsst::afw::geom::ellipses::BaseCore::ParameterVector &,
     lsst::afw::geom::ellipses::BaseCore::ParameterVector const &
{
    $result = Py_BuildValue("ddd", $1->x(), $1->y(), $1->z());
}


%ignore lsst::afw::geom::ellipses::BaseEllipse::operator[];
%ignore lsst::afw::geom::ellipses::BaseEllipse::getVector;
%ignore lsst::afw::geom::ellipses::BaseEllipse::setVector;
%ignore lsst::afw::geom::ellipses::BaseCore::operator[];
%ignore lsst::afw::geom::ellipses::BaseCore::getVector;
%ignore lsst::afw::geom::ellipses::BaseCore::setVector;

SWIG_SHARED_PTR(BaseEllipse, lsst::afw::geom::ellipses::BaseEllipse);
SWIG_SHARED_PTR(BaseCore, lsst::afw::geom::ellipses::BaseCore);

%include "lsst/afw/geom/ellipses/BaseEllipse.h"

%extend lsst::afw::geom::ellipses::BaseCore {
    double get(int i) {return self->operator[](i); }
    void set(int i, double value) {
        self->operator[](i) = value;
    }
    %pythoncode {
    def __getitem__(self, k):
        return self.get(k)
    
    def __setitem__(self, k, v):
        self.set(k, v)
    }
}

%extend lsst::afw::geom::ellipses::BaseEllipse {
    double get(int i) {return self->operator[](i); }
    void set(int i, double value) {
        self->operator[](i) = value;
    }
    
    void setCore(lsst::afw::geom::ellipses::BaseCore const & core) {
    	self->getCore() = core;
    }     
    
    %pythoncode {
    def __getitem__(self, k):
        return self.get(k)
    
    def __setitem__(self, k, v):
        self.set(k, v)
    }
}

SWIG_SHARED_PTR_DERIVED(Ellipse, lsst::afw::geom::ellipses::BaseEllipse, lsst::afw::geom::ellipses::AxesEllipse);
SWIG_SHARED_PTR_DERIVED(Ellipse, lsst::afw::geom::ellipses::BaseCore, lsst::afw::geom::ellipses::Axes);
SWIG_SHARED_PTR_DERIVED(Ellipse, lsst::afw::geom::ellipses::BaseEllipse, lsst::afw::geom::ellipses::DistortionEllipse);
SWIG_SHARED_PTR_DERIVED(Ellipse, lsst::afw::geom::ellipses::BaseCore, lsst::afw::geom::ellipses::Distortion);
SWIG_SHARED_PTR_DERIVED(Ellipse, lsst::afw::geom::ellipses::BaseEllipse, lsst::afw::geom::ellipses::LogShearEllipse);
SWIG_SHARED_PTR_DERIVED(Ellipse, lsst::afw::geom::ellipses::BaseCore, lsst::afw::geom::ellipses::LogShear);
SWIG_SHARED_PTR_DERIVED(Ellipse, lsst::afw::geom::ellipses::BaseEllipse, lsst::afw::geom::ellipses::QuadrupoleEllipse);
SWIG_SHARED_PTR_DERIVED(Ellipse, lsst::afw::geom::ellipses::BaseCore, lsst::afw::geom::ellipses::Quadrupole);

%feature("notabstract") lsst::afw::geom::ellipses::Axes;
%feature("notabstract") lsst::afw::geom::ellipses::Distortion;
%feature("notabstract") lsst::afw::geom::ellipses::LogShear;
%feature("notabstract") lsst::afw::geom::ellipses::Quadrupole;

%include "lsst/afw/geom/ellipses/Axes.h"
%include "lsst/afw/geom/ellipses/Distortion.h"
%include "lsst/afw/geom/ellipses/Quadrupole.h"
%include "lsst/afw/geom/ellipses/LogShear.h"

%define %core(NAME...)

%extend lsst::afw::geom::ellipses::NAME {
    void set(lsst::afw::geom::ellipses::Core const & other) {
        self->operator=(other);
    }
}
%enddef

%core(Axes);
%core(Distortion);
%core(LogShear);
%core(Quadrupole);
