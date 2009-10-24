%include "affine_transform.i"

%{
#include "Eigen/Core"


#include "lsst/afw/math/ellipses/Core.h"
#include "lsst/afw/math/ellipses/Ellipse.h"
#include "lsst/afw/math/ellipses/Axes.h"
#include "lsst/afw/math/ellipses/Distortion.h"
#include "lsst/afw/math/ellipses/Quadrupole.h"
#include "lsst/afw/math/ellipses/LogShear.h"
%}

%include "std_complex.i"

//define mappings from Eigen::Vectors to tuples
%typemap(out) Eigen::Vector3d &, Eigen::Vector3d const & {
	$result = Py_BuildValue("ddd", $1->x(), $1->y(), $1->z());
}


%ignore lsst::afw::math::ellipses::Core::operator[];
%ignore lsst::afw::math::ellipses::Ellipse::operator[];
%ignore lsst::afw::math::ellipses::Core::getScalingDerivative;
%ignore lsst::afw::math::ellipses::Core::setVector;
%ignore lsst::afw::math::ellipses::Core::makeEllipse;

%include "lsst/afw/math/ellipses/Core.h"

%extend lsst::afw::math::ellipses::Core {
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

%include "lsst/afw/math/ellipses/Ellipse.h"

%extend lsst::afw::math::ellipses::Ellipse {
    double get(int i) {return self->operator[](i); }
    void set(int i, double value) {
        self->operator[](i) = value;
    }
    
    void setCore(lsst::afw::math::ellipses::Core const & core) {
    	self->getCore() = core;
    }     
    
    %pythoncode {
    def __getitem__(self, k):
        return self.get(k)
    
    def __setitem__(self, k, v):
        self.set(k, v)
    }
}

%feature("notabstract") lsst::afw::math::ellipses::Axes;
%feature("notabstract") lsst::afw::math::ellipses::Distortion;
%feature("notabstract") lsst::afw::math::ellipses::LogShear;
%feature("notabstract") lsst::afw::math::ellipses::Quadrupole;

%include "lsst/afw/math/ellipses/Axes.h"
%include "lsst/afw/math/ellipses/Distortion.h"
%include "lsst/afw/math/ellipses/Quadrupole.h"
%include "lsst/afw/math/ellipses/LogShear.h"


%define %core(NAME...)

%extend lsst::afw::math::ellipses::NAME {
	void set(lsst::afw::math::ellipses::Core const & other) {
		self->operator=(other);
	}
    %pythoncode {      	       	    	
    def clone(self):
        return NAME(self)
    
    def makeEllipse(self):
        return NAME##Ellipse()
    }
}
%enddef

%core(Axes);
%core(Distortion);
%core(LogShear);
%core(Quadrupole);
