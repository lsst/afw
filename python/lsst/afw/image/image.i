// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
%{
#   include "lsst/afw/image/Image.h"
#   include "lsst/afw/image/ImagePca.h"
%}

//
// Must go Before the %include
//
%define %imagePtr(PIXEL_TYPE...)
%shared_ptr(lsst::afw::image::ImageBase<PIXEL_TYPE>);
%shared_ptr(lsst::afw::image::Image<PIXEL_TYPE>);
%shared_ptr(lsst::afw::image::DecoratedImage<PIXEL_TYPE>);
%declareNumPyConverters(lsst::afw::image::ImageBase<PIXEL_TYPE>::Array);
%declareNumPyConverters(lsst::afw::image::ImageBase<PIXEL_TYPE>::ConstArray);
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
%template(NAME##PcaM##TYPE) lsst::afw::image::ImagePca<lsst::afw::image::MaskedImage<PIXEL_TYPE> >;

%template(innerProduct) lsst::afw::image::innerProduct<lsst::afw::image::Image<PIXEL_TYPE>,
                                                       lsst::afw::image::Image<PIXEL_TYPE>  >;

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
        self->operator()(x, y, lsst::afw::image::CheckIndices(true)) = val;
    }

    PIXEL_TYPE get(int x, int y) {
        return self->operator()(x, y, lsst::afw::image::CheckIndices(true));
    }

    void set0(int x, int y, double val) {
        self->set0(x, y, val, lsst::afw::image::CheckIndices(true));
    }
    PIXEL_TYPE get0(int x, int y) {
        PIXEL_TYPE p = self->get0(x, y, lsst::afw::image::CheckIndices(true));
        return p;
    }

    %pythoncode %{
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

# support "__from__ future import division" in Python 2; not needed for Python 3
__itruediv__ = __idiv__

    %}
}
%defineClone(NAME##TYPE, lsst::afw::image::Image, PIXEL_TYPE);
%supportSlicing(lsst::afw::image::Image, PIXEL_TYPE);
%fits_reduce(lsst::afw::image::Image<PIXEL_TYPE>);
%enddef

/************************************************************************************************************/

%ignore lsst::afw::image::ImageBase::operator();
%ignore lsst::afw::image::ImageBase::get0;
%ignore lsst::afw::image::ImageBase::set0;
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

%imagePtr(boost::uint16_t);
%imagePtr(boost::uint64_t);
%imagePtr(int);
%imagePtr(float);
%imagePtr(double);

%include "lsst/afw/image/Utils.h"
%include "lsst/afw/image/Image.h"
%include "lsst/afw/image/ImagePca.h"
%include "lsst/afw/image/Mask.h"

%image(Image, U, boost::uint16_t);
%image(Image, L, boost::uint64_t);
%image(Image, I, int);
%image(Image, F, float);
%image(Image, D, double);

%template(vectorBBox) std::vector<lsst::afw::geom::BoxI>;         

%extend lsst::afw::image::Image<boost::uint16_t> {
    %newobject convertF;
    lsst::afw::image::Image<float> convertF() {
       return lsst::afw::image::Image<float>(*self, true);
    }
    %pythoncode %{
    def convertFloat(self, *args):
        """Alias for convertF"""

        return self.convertF(*args)
    %}
}

%extend lsst::afw::image::Image<boost::uint64_t> {
    %newobject convertD;
    lsst::afw::image::Image<double> convertD() {
       return lsst::afw::image::Image<double>(*self, true);
    }
    %pythoncode %{
    def convertDouble(self, *args):
        """Alias for convertD"""

        return self.convertD(*args)
    %}
}

%extend lsst::afw::image::Image<double> {
    %newobject convertF;
    lsst::afw::image::Image<float> convertF() {
       return lsst::afw::image::Image<float>(*self, true);
    }
    %pythoncode %{
    def convertFloat(self, *args):
        """Alias for convertF"""

        return self.convertF(*args)
    %}
}

%extend lsst::afw::image::Image<float> {
    %newobject convertD;
    lsst::afw::image::Image<double> convertD() {
        return lsst::afw::image::Image<double>(*self, true);
    }
    %newobject convertU;
    lsst::afw::image::Image<boost::uint16_t> convertU() {
        return lsst::afw::image::Image<boost::uint16_t>(*self, true);
    }
    %pythoncode %{
    def convertU16(self, *args):
        """Alias for convertU"""

        return self.convertU(*args)
    %}
}

