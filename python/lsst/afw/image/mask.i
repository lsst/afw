// -*- lsst-c++ -*-
%{
#   include "lsst/afw/image/Mask.h"
%}

%ignore lsst::afw::image::Mask::operator();

/************************************************************************************************************/
//
// Must go Before the %include
//
// N.b. assumes that the corresponding image has been declared to swig; otherwise
// you'll need something like
//    SWIG_SHARED_PTR(NAME##TYPE##BasePtr, lsst::afw::image::ImageBase<PIXEL_TYPE>);
//
//
%define %maskPtr(NAME, TYPE, PIXEL_TYPE...)
SWIG_SHARED_PTR_DERIVED(NAME##TYPE##Ptr, lsst::afw::image::ImageBase<PIXEL_TYPE>, lsst::afw::image::Mask<PIXEL_TYPE>);
%enddef

//
// Must go After the %include
//
%define %mask(NAME, TYPE, PIXEL_TYPE...)
%template(NAME##TYPE) lsst::afw::image::Mask<PIXEL_TYPE>;

%extend lsst::afw::image::Mask<PIXEL_TYPE> {
    /**
     * Set an image to the value val
     */
    void set(PIXEL_TYPE val) {
        *self = val;
    }

    /**
     * Set pixel (x,y) to val
     */
    void set(int x, int y, double val) {
        self->operator()(x, y) = val;
    }

    PIXEL_TYPE get(int x, int y) const {
        return self->operator()(x, y);
    }

    bool get(int x, int y, int plane) const {
        return self->operator()(x, y, plane);
    }
    %pythoncode {
    #
    # Deal with incorrect swig wrappers for C++ "void operator op=()"
    #
    def __ilshift__(*args):
        """__ilshift__(self, NAME## Type src) -> self"""
        _imageLib.Image##TYPE##Base___ilshift__(*args)
        return args[0]

    def __ior__(*args):
        """__ior__(self, NAME##Type src) -> self"""
        _imageLib.NAME##TYPE##___ior__(*args)
        return args[0]

    def __iand__(*args):
        """__iand__(self, NAME##TYPE src) -> self"""
        _imageLib.NAME##TYPE##___iand__(*args)
        return args[0]
    }
}
%enddef

/************************************************************************************************************/

%maskPtr(Mask, U, boost::uint16_t);

%include "lsst/afw/image/Mask.h"

%mask(Mask, U, boost::uint16_t);

