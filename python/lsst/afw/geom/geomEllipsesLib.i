// -*- lsst-c++ -*-
%define geomEllipsesLib_DOCSTRING
"
Python interface to lsst::afw::geom::ellipses classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.geom.ellipses",docstring=geomEllipsesLib_DOCSTRING) geomEllipsesLib

%{
#include "Eigen/Core"

#include "lsst/afw/geom/ellipses.h"
%}

%include "lsst/p_lsstSwig.i"
%include "std_complex.i"

%pythoncode %{
import lsst.utils

def version(HeadURL = r"$HeadURL: svn+ssh://svn.lsstcorp.org/DMS/afw/trunk/python/lsst/afw/geom/geomEllipsesLib.i $"):
    """Return a version given a HeadURL string. If a different version is setup, return that too"""

    version_svn = lsst.utils.guessSvnVersion(HeadURL)

    try:
        import eups
    except ImportError:
        return version_svn
    else:
        try:
            version_eups = eups.setup("afw")
        except AttributeError:
            return version_svn

    if version_eups == version_svn:
        return version_svn
    else:
        return "%s (setup: %s)" % (version_svn, version_eups)

%}

%import "lsst/afw/geom/geomLib.i"

%lsst_exceptions();

//define mappings from Eigen::Vectors to tuples
// %typemap(out)
//      ellipses::BaseCore::ParameterVector &,
//      ellipses::BaseCore::ParameterVector const &
// {
//     $result = Py_BuildValue("ddd", $1->x(), $1->y(), $1->z());
// }
// %typemap(out)
//      ellipses::BaseEllipse::ParameterVector &,
//      ellipses::BaseEllipse::ParameterVector const &
// {
//     $result = Py_BuildValue("ddddd", $1[0], $1[1], $1[2], $1[3], $1[4]);
// }


%ignore lsst::afw::geom::ellipses::BaseEllipse::operator[];
%ignore lsst::afw::geom::ellipses::BaseEllipse::getVector;
%ignore lsst::afw::geom::ellipses::BaseEllipse::setVector;
%ignore lsst::afw::geom::ellipses::BaseCore::operator[];
%ignore lsst::afw::geom::ellipses::BaseCore::getVector;
%ignore lsst::afw::geom::ellipses::BaseCore::setVector;

%rename(set) lsst::afw::geom::ellipses::BaseEllipse::operator=;
%rename(set) lsst::afw::geom::ellipses::BaseCore::operator=;

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

/*
SWIG_SHARED_PTR_DERIVED(Ellipse, ellipses::BaseEllipse, ellipses::AxesEllipse);
SWIG_SHARED_PTR_DERIVED(Ellipse, ellipses::BaseCore, ellipses::Axes);
SWIG_SHARED_PTR_DERIVED(Ellipse, ellipses::BaseEllipse, ellipses::DistortionEllipse);
SWIG_SHARED_PTR_DERIVED(Ellipse, ellipses::BaseCore, ellipses::Distortion);
SWIG_SHARED_PTR_DERIVED(Ellipse, ellipses::BaseEllipse, ellipses::LogShearEllipse);
SWIG_SHARED_PTR_DERIVED(Ellipse, ellipses::BaseCore, ellipses::LogShear);
SWIG_SHARED_PTR_DERIVED(Ellipse, ellipses::BaseEllipse, ellipses::QuadrupoleEllipse);
SWIG_SHARED_PTR_DERIVED(Ellipse, ellipses::BaseCore, ellipses::Quadrupole);

%include "lsst/afw/geom/ellipses/EllipseImpl.h"
%template(AxesEllipseImpl) ellipses::detail::EllipseImpl<ellipses::Axes,ellipses::AxesEllipse>;


%feature("notabstract") ellipses::Axes;
%feature("notabstract") ellipses::Distortion;
%feature("notabstract") ellipses::LogShear;
%feature("notabstract") ellipses::Quadrupole;

%include "lsst/afw/geom/ellipses/Axes.h"
%include "lsst/afw/geom/ellipses/Distortion.h"
%include "lsst/afw/geom/ellipses/Quadrupole.h"
%include "lsst/afw/geom/ellipses/LogShear.h"

%define %core(NAME...)

%extend ellipses::NAME {
    void set(ellipses::Core const & other) {
        self->operator=(other);
    }
}
%enddef

%core(Axes);
%core(Distortion);
%core(LogShear);
%core(Quadrupole);
*/
