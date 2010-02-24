// -*- lsst-c++ -*-
%{
#   include "lsst/afw/image/Image.h"
#   include "lsst/afw/image/ImagePca.h"
%}

%ignore lsst::afw::image::ImageBase::operator();

//
// Must go Before the %include
//
%define %imagePtr(NAME, TYPE, PIXEL_TYPE...)
SWIG_SHARED_PTR_DERIVED(NAME##TYPE##Base, lsst::daf::data::LsstBase, lsst::afw::image::ImageBase<PIXEL_TYPE>);
SWIG_SHARED_PTR_DERIVED(NAME##TYPE, lsst::afw::image::ImageBase<PIXEL_TYPE>, lsst::afw::image::Image<PIXEL_TYPE>);
SWIG_SHARED_PTR(Decorated##NAME##TYPE, lsst::afw::image::DecoratedImage<PIXEL_TYPE>);
%enddef

//
// Must go After the %include
//
%define %image(NAME, TYPE, PIXEL_TYPE...)
%template(NAME##TYPE##Base) lsst::afw::image::ImageBase<PIXEL_TYPE>;
%template(NAME##TYPE) lsst::afw::image::Image<PIXEL_TYPE>;
%template(Decorated##NAME##TYPE) lsst::afw::image::DecoratedImage<PIXEL_TYPE>;
%lsst_persistable(lsst::afw::image::Image<PIXEL_TYPE>);
%lsst_persistable(lsst::afw::image::DecoratedImage<PIXEL_TYPE>);

%template(vector##NAME##TYPE) std::vector<boost::shared_ptr<lsst::afw::image::Image<PIXEL_TYPE> > >;
%template(NAME##Pca##TYPE) lsst::afw::image::ImagePca<lsst::afw::image::Image<PIXEL_TYPE> >;

%template(innerProduct) lsst::afw::image::innerProduct<lsst::afw::image::Image<PIXEL_TYPE> >;

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
    def Factory(self, *args):
        """Return an Image class of this type
        
        A synonym for the attribute __class__
        """
        return NAME##TYPE(*args)
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


%define %mimage(NAME, TYPE, PIXEL_TYPE...)
%template(vector##NAME##TYPE) std::vector<boost::shared_ptr<lsst::afw::image::MaskedImage<PIXEL_TYPE> > >;
%enddef

/************************************************************************************************************/

%ignore lsst::afw::image::ImageBase::swap;
%ignore lsst::afw::image::ImageBase::begin;
%ignore lsst::afw::image::ImageBase::end;
%ignore lsst::afw::image::ImageBase::rbegin;
%ignore lsst::afw::image::ImageBase::rend;
%ignore lsst::afw::image::ImageBase::at;
%ignore lsst::afw::image::ImageBase::row_begin;
%ignore lsst::afw::image::ImageBase::row_end;
%ignore lsst::afw::image::ImageBase::x_at;
%ignore lsst::afw::image::ImageBase::col_begin;
%ignore lsst::afw::image::ImageBase::col_end;
%ignore lsst::afw::image::ImageBase::y_at;
%ignore lsst::afw::image::ImageBase::xy_at;

SWIG_SHARED_PTR_DERIVED(Wcs, lsst::daf::data::LsstBase, lsst::afw::image::Wcs);
SWIG_SHARED_PTR_DERIVED(Wcs, lsst::daf::data::LsstBase, lsst::afw::image::TanWcs);
%imagePtr(Image, U, boost::uint16_t);
%imagePtr(Image, I, int);
%imagePtr(Image, F, float);
%imagePtr(Image, D, double);

%include "lsst/afw/image/Utils.h"
%include "lsst/afw/image/Image.h"
%include "lsst/afw/image/ImagePca.h"

%image(Image, U, boost::uint16_t);
%image(Image, I, int);
%image(Image, F, float);
%image(Image, D, double);

%mimage(MaskedImage, U, boost::uint16_t);
%mimage(MaskedImage, I, int);
%mimage(MaskedImage, F, float);
%mimage(MaskedImage, D, double);

%template(vectorBBox) std::vector<lsst::afw::image::BBox>;         

%extend lsst::afw::image::Image<boost::uint16_t> {
    %newobject convertFloat;
    lsst::afw::image::Image<float> convertFloat() {
       return lsst::afw::image::Image<float>(*self, true);
    }
}

%extend lsst::afw::image::Image<double> {
    %newobject convertFloat;
    lsst::afw::image::Image<float> convertFloat() {
       return lsst::afw::image::Image<float>(*self, true);
    }
}

%extend lsst::afw::image::Image<float> {
   %newobject convertU16;
   lsst::afw::image::Image<boost::uint16_t> convertU16() {
       return lsst::afw::image::Image<boost::uint16_t>(*self, true);
   }
}
