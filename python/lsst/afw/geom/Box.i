
%{
#include "lsst/afw/geom/Box.h"
%}

%rename(set) lsst::afw::geom::BoxI::operator=;
%rename(__eq__) lsst::afw::geom::BoxI::operator==;
%rename(__ne__) lsst::afw::geom::BoxI::operator!=;
%copyctor lsst::afw::geom::BoxI;
%rename(set) lsst::afw::geom::BoxD::operator=;
%rename(__eq__) lsst::afw::geom::BoxD::operator==;
%rename(__ne__) lsst::afw::geom::BoxD::operator!=;
%copyctor lsst::afw::geom::BoxD;

%include "lsst/afw/geom/Box.h"

%extend lsst::afw::geom::BoxI {             
    %pythoncode {
    def __repr__(self):
        return "BoxI(%r, %r)" % (self.getMin(), self.getDimensions())
             
    def __str__(self):
        return "BoxI(%s, %s)" % (self.getMin(), self.getMax())
    }
}

%extend lsst::afw::geom::BoxD {             
    %pythoncode {
    def __repr__(self):
        return "BoxD(%r, %r)" % (self.getMin(), self.getDimensions())
             
    def __str__(self):
        return "BoxD(%s, %s)" % (self.getMin(), self.getMax())
    }
}
