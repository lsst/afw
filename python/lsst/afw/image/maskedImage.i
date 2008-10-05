// -*- lsst-c++ -*-
%{
#   include "lsst/afw/image/MaskedImage.h"
%}

/************************************************************************************************************/
//
// Must go Before the %include
//
// N.b. assumes that the corresponding image has been declared to swig; otherwise
// you'll need something like
//    SWIG_SHARED_PTR(NAME##BasePtr, lsst::afw::image::ImageBase<PIXEL_TYPE>);
//
//
%define %maskedImagePtr(NAME, TYPE, PIXEL_TYPE...)
SWIG_SHARED_PTR(NAME##TYPE##Ptr, lsst::afw::image::MaskedImage<PIXEL_TYPE>);
%enddef

//
// Must go After the %include
//
%define %maskedImage(NAME, TYPE, PIXEL_TYPE...)
%template(NAME##TYPE) lsst::afw::image::MaskedImage<PIXEL_TYPE>;

%extend lsst::afw::image::MaskedImage<PIXEL_TYPE> {
    %pythoncode {
    def set(self, x, row, values):
        """Set the point (x, row) to a triple (value, mask, variance)"""

        try:
            return (self.getImage().set(x, row, values[0]),
                    self.getMask().set(x, row, values[1]),
                    self.getVariance().set(x, row, values[2]))
        except TypeError:
            return (self.getImage().set(x),
                    self.getMask().set(row),
                    self.getVariance().set(values))

    def get(self, x, row):
        """Return a triple (value, mask, variance) at the point (x, row)"""
        return (self.getImage().get(x, row),
                self.getMask().get(x, row),
                self.getVariance().get(x, row))

    #
    # Deal with incorrect swig wrappers for C++ "void operator op=()"
    #
    def __ilshift__(*args):
        """__ilshift__(self, NAME##TYPE src) -> self"""
        _imageLib.MaskedImage##TYPE##___ilshift__(*args)
        return args[0]

    def __ior__(*args):
        """__ior__(self, NAME##TYPE src) -> self"""
        _imageLib.NAME##TYPE##___ior__(*args)
        return args[0]

    def __iand__(*args):
        """__iand__(self, NAME##TYPE src) -> self"""
        _imageLib.NAME##TYPE##___iand__(*args)
        return args[0]

    def __iadd__(*args):
        """
        __iadd__(self, float scalar) -> self
        __iadd__(self, NAME##TYPE inputImage) -> self
        """
        _imageLib.NAME##TYPE##___iadd__(*args)
        return args[0]

    def __isub__(*args):
        """
        __isub__(self, float scalar)
        __isub__(self, NAME##TYPE inputImage)
        """
        _imageLib.NAME##TYPE##___isub__(*args)
        return args[0]
    

    def __imul__(*args):
        """
        __imul__(self, float scalar)
        __imul__(self, NAME##TYPE inputImage)
        """
        _imageLib.NAME##TYPE##___imul__(*args)
        return args[0]

    def __idiv__(*args):
        """
        __idiv__(self, float scalar)
        __idiv__(self, NAME##TYPE inputImage)
        """
        _imageLib.NAME##TYPE##___idiv__(*args)
        return args[0]
    }
}
%enddef

/************************************************************************************************************/

%ignore lsst::afw::image::MaskedImage::operator();
%ignore lsst::afw::image::MaskedImage::begin;
%ignore lsst::afw::image::MaskedImage::end;
%ignore lsst::afw::image::MaskedImage::at;
%ignore lsst::afw::image::MaskedImage::rbegin;
%ignore lsst::afw::image::MaskedImage::rend;
%ignore lsst::afw::image::MaskedImage::col_begin;
%ignore lsst::afw::image::MaskedImage::col_end;
%ignore lsst::afw::image::MaskedImage::y_at;
%ignore lsst::afw::image::MaskedImage::row_begin;
%ignore lsst::afw::image::MaskedImage::row_end;
%ignore lsst::afw::image::MaskedImage::x_at;
%ignore lsst::afw::image::MaskedImage::xy_at;

%maskedImagePtr(MaskedImage, F, float);

%include "lsst/afw/image/MaskedImage.h"

%maskedImage(MaskedImage, F, float);

