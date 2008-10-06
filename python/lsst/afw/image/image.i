// -*- lsst-c++ -*-
%{
#   include "lsst/afw/image/Image.h"
%}

%ignore lsst::afw::image::ImageBase::operator();

//
// Must go Before the %include
//
%define %imagePtr(NAME, TYPE, PIXEL_TYPE...)
SWIG_SHARED_PTR(NAME##TYPE##BasePtr, lsst::afw::image::ImageBase<PIXEL_TYPE>);
SWIG_SHARED_PTR_DERIVED(NAME##TYPE##Ptr, lsst::afw::image::ImageBase<PIXEL_TYPE>, lsst::afw::image::Image<PIXEL_TYPE>);
SWIG_SHARED_PTR(Decorated##NAME##TYPE##Ptr, lsst::afw::image::DecoratedImage<PIXEL_TYPE>);

%template(NAME##TYPE##Ptr) boost::shared_ptr<lsst::afw::image::Image<PIXEL_TYPE> >;
%enddef

//
// Must go After the %include
//
%define %image(NAME, TYPE, PIXEL_TYPE...)
%template(NAME##TYPE##Base) lsst::afw::image::ImageBase<PIXEL_TYPE>;
%template(NAME##TYPE) lsst::afw::image::Image<PIXEL_TYPE>;
%template(Decorated##NAME##TYPE) lsst::afw::image::DecoratedImage<PIXEL_TYPE>;

//%lsst_persistable_shared_ptr(NAME##TYPE##Ptr, lsst::afw::image::Image<PIXEL_TYPE>)

%extend lsst::afw::image::Image<PIXEL_TYPE> {
    /**
     * Set an image to the value val
     */
    void set(double val) {
        *self = val;
    }
    /**
     * Set pixel (x,y) to val
     */
    void set(int x, int y, double val) {
        self->operator()(x, y) = val;
    }

    PIXEL_TYPE get(int x, int y) {
        return self->operator()(x, y);
    }

    %pythoncode {
    #
    # Deal with incorrect swig wrappers for C++ "void operator op=()"
    #
    def __ilshift__(*args):
        """__ilshift__(self, NAME src) -> self"""
        _imageLib.NAME##TYPE##Base##___ilshift__(*args)
        return args[0]

    def __iadd__(*args):
        """
        __iadd__(self, float scalar) -> self
        __iadd__(self, NAME inputImage) -> self
        """
        _imageLib.NAME##TYPE##___iadd__(*args)
        return args[0]

    def __isub__(*args):
        """
        __isub__(self, float scalar)
        __isub__(self, NAME inputImage)
        """
        _imageLib.NAME##TYPE##___isub__(*args)
        return args[0]
    

    def __imul__(*args):
        """
        __imul__(self, float scalar)
        __imul__(self, NAME inputImage)
        """
        _imageLib.NAME##TYPE##___imul__(*args)
        return args[0]

    def __idiv__(*args):
        """
        __idiv__(self, float scalar)
        __idiv__(self, NAME inputImage)
        """
        _imageLib.NAME##TYPE##___idiv__(*args)
        return args[0]
    }
}
%enddef

/************************************************************************************************************/

%ignore swap;

%imagePtr(Image, U, boost::uint16_t);
%imagePtr(Image, F, float);

%include "lsst/afw/image/Image.h"

%image(Image, U, boost::uint16_t);
%image(Image, I, int);
%image(Image, F, float);
%image(Image, D, double);

%extend lsst::afw::image::Image<boost::uint16_t> {
    %newobject convertFloat;
    lsst::afw::image::Image<float> convertFloat() {
       return Image<float>(*self, true);
    }
}

%extend lsst::afw::image::Image<float> {
   %newobject convertU16;
   lsst::afw::image::Image<boost::uint16_t> convertU16() {
       return Image<boost::uint16_t>(*self, true);
   }
}
