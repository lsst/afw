
%{
#include "lsst/afw/geom/Box.h"
%}

%rename(set) lsst::afw::geom::Box2I::operator=;
%rename(__eq__) lsst::afw::geom::Box2I::operator==;
%rename(__ne__) lsst::afw::geom::Box2I::operator!=;
%copyctor lsst::afw::geom::Box2I;
%rename(set) lsst::afw::geom::Box2D::operator=;
%rename(__eq__) lsst::afw::geom::Box2D::operator==;
%rename(__ne__) lsst::afw::geom::Box2D::operator!=;
%copyctor lsst::afw::geom::Box2D;

%include "lsst/afw/geom/Box.h"

%extend lsst::afw::geom::Box2I {             
    %pythoncode {
    def __repr__(self):
        return "Box2I(%r, %r)" % (self.getMin(), self.getDimensions())
             
    def __str__(self):
        return "Box2I(%s, %s)" % (self.getMin(), self.getMax())
    }
}

%extend lsst::afw::geom::Box2D {             
    %pythoncode {
    def __repr__(self):
        return "Box2D(%r, %r)" % (self.getMin(), self.getDimensions())
             
    def __str__(self):
        return "Box2D(%s, %s)" % (self.getMin(), self.getMax())
    }
}
