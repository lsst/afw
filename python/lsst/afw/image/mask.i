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
#   include "lsst/afw/image/Mask.h"
%}

%ignore lsst::afw::image::Mask::operator();

/************************************************************************************************************/
//
// Must go Before the %include
//
// N.b. assumes that the corresponding image has been declared to swig; otherwise
// you'll need something like
//    SWIG_SHARED_PTR(NAME##TYPE##Base, lsst::afw::image::ImageBase<PIXEL_TYPE>);
//
//
%define %maskPtr(NAME, TYPE, PIXEL_TYPE...)
SWIG_SHARED_PTR_DERIVED(NAME##TYPE, lsst::afw::image::ImageBase<PIXEL_TYPE>, lsst::afw::image::Mask<PIXEL_TYPE>);
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
        self->operator()(x, y, true) = val;
    }

    PIXEL_TYPE get(int x, int y) const {
        return self->operator()(x, y, true);
    }

    bool get(int x, int y, int plane) const {
        return self->operator()(x, y, plane);
    }
    %pythoncode {
    def Factory(self, *args):
        """Return a Mask of this type"""
        return NAME##TYPE(*args)
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

    def __ixor__(*args):
        """__ixor__(self, NAME##TYPE src) -> self"""
        _imageLib.NAME##TYPE##___ixor__(*args)
        return args[0]
    }
}
%enddef

/************************************************************************************************************/
%apply int { unsigned short };

%maskPtr(Mask, U, boost::uint16_t);

%include "lsst/afw/image/Mask.h"

%mask(Mask, U, boost::uint16_t);
