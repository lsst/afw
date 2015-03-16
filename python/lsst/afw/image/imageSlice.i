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
 
//

%{
#include "lsst/afw/image/ImageSlice.h"
%}

%include "lsst/afw/image/ImageSlice.h"

%define %slice(NAME, TYPE, PIXEL_TYPE...)

%shared_ptr(lsst::afw::image::ImageSlice<PIXEL_TYPE>);
%template(NAME##TYPE) lsst::afw::image::ImageSlice<PIXEL_TYPE>;
%template(NAME##TYPE##___add__) lsst::afw::image::operator+<PIXEL_TYPE>;
%template(NAME##TYPE##___sub__) lsst::afw::image::operator-<PIXEL_TYPE>;
%template(NAME##TYPE##___mul__) lsst::afw::image::operator*<PIXEL_TYPE>;
%template(NAME##TYPE##___div__) lsst::afw::image::operator/<PIXEL_TYPE>;

%extend lsst::afw::image::ImageSlice<PIXEL_TYPE> {


    %pythoncode %{
         
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

# support "__from__ future import division" in Python 2 (not needed for Python 3)
__truediv__ = __div__
    %}

}


%extend lsst::afw::image::Image<PIXEL_TYPE> {


    %pythoncode %{
         
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

# support "__from__ future import division" in Python 2 (not needed for Python 3)
__truediv__ = __div__
    %}

}


%enddef


 //%slice(ImageSlice, U, boost::uint16_t);
 //%slice(ImageSlice, I, int);
%slice(ImageSlice, F, float);
%slice(ImageSlice, D, double);
