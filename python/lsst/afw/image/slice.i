// -*- lsst-c++ -*-
//

%{
#include "lsst/afw/image/Slice.h"
%}

%include "lsst/afw/image/Slice.h"

%define %slice(NAME, TYPE, PIXEL_TYPE...)

SWIG_SHARED_PTR_DERIVED(NAME##TYPE, lsst::afw::image::Image<PIXEL_TYPE>, lsst::afw::image::Slice<PIXEL_TYPE>);
%template(NAME##TYPE) lsst::afw::image::Slice<PIXEL_TYPE>;
%template(NAME##TYPE##___add__) lsst::afw::image::operator+<PIXEL_TYPE>;
%template(NAME##TYPE##___sub__) lsst::afw::image::operator-<PIXEL_TYPE>;
%template(NAME##TYPE##___mul__) lsst::afw::image::operator*<PIXEL_TYPE>;
%template(NAME##TYPE##___div__) lsst::afw::image::operator/<PIXEL_TYPE>;

%extend lsst::afw::image::Slice<PIXEL_TYPE> {


    %pythoncode {
         
    #
    # Deal with incorrect swig wrappers for C++ "void operator op=()"
    #
    def __add__(*args):
        """
        __add__(self, float scalar) -> self
        __add__(self, NAME inputImage) -> self
        """
        return _imageLib.__add__(*args)

    def __sub__(*args):
        """
        __sub__(self, float scalar)
        __sub__(self, NAME inputImage)
        """
        return _imageLib.__sub__(*args)

    def __mul__(*args):
        """
        __mul__(self, float scalar)
        __mul__(self, NAME inputImage)
        """
        return _imageLib.__mul__(*args)

    def __div__(*args):
        """
        __div__(self, float scalar)
        __div__(self, NAME inputImage)
        """
        return _imageLib.__div__(*args)
    }

}


%extend lsst::afw::image::Image<PIXEL_TYPE> {


    %pythoncode {
         
    #
    # Deal with incorrect swig wrappers for C++ "void operator op=()"
    #
    def __add__(*args):
        """
        __add__(self, float scalar) -> self
        __add__(self, NAME inputImage) -> self
        """
        return _imageLib.__add__(*args)

    def __sub__(*args):
        """
        __sub__(self, float scalar)
        __sub__(self, NAME inputImage)
        """
        return _imageLib.__sub__(*args)

    def __mul__(*args):
        """
        __mul__(self, float scalar)
        __mul__(self, NAME inputImage)
        """
        return _imageLib.__mul__(*args)

    def __div__(*args):
        """
        __div__(self, float scalar)
        __div__(self, NAME inputImage)
        """
        return _imageLib.__div__(*args)
    }

}


%enddef


 //%slice(Slice, U, boost::uint16_t);
 //%slice(Slice, I, int);
%slice(Slice, F, float);
%slice(Slice, D, double);
