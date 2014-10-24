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
#include "boost/shared_ptr.hpp"
#include "lsst/afw/math/Background.h"
%}

%shared_ptr(lsst::afw::math::BackgroundControl);
%shared_ptr(lsst::afw::math::Background);
%shared_ptr(lsst::afw::math::BackgroundMI);

%include "lsst/afw/math/Background.h"

%inline %{
   PTR(lsst::afw::math::BackgroundMI)
   cast_BackgroundMI(PTR(lsst::afw::math::Background) bback) {
        return boost::dynamic_pointer_cast<lsst::afw::math::BackgroundMI>(bback);
    }
%}
%extend lsst::afw::math::Background {
    %pythoncode %{
#
# Deal with incorrect swig wrappers for C++ "void operator op=()"
#
def __iadd__(*args):
    """
    __iadd__(self, float scalar) -> self
    """
    _mathLib.Background___iadd__(*args) # clears thisown as it things args[0] is returned
    args[0].thisown = True
    return args[0]

def __isub__(*args):
    """
    __isub__(self, float scalar) -> self
    """
    _mathLib.Background___isub__(*args) # clears thisown as it things args[0] is returned
    args[0].thisown = True
    return args[0]
    %}
}

%extend lsst::afw::math::BackgroundMI {
    %pythoncode %{
#
# Deal with incorrect swig wrappers for C++ "void operator op=()"
#
def __iadd__(*args):
    """
    __iadd__(self, float scalar) -> self
    """
    _mathLib.BackgroundMI___iadd__(*args) # clears thisown as it things args[0] is returned
    args[0].thisown = True
    return args[0]

def __isub__(*args):
    """
    __isub__(self, float scalar) -> self
    """
    _mathLib.BackgroundMI___isub__(*args) # clears thisown as it things args[0] is returned
    args[0].thisown = True
    return args[0]

def __reduce__(self):
    """Pickling"""
    return self.__class__, (self.getImageBBox(), self.getStatsImage())
    %}
}

%define %declareBack(PIXTYPE, SUFFIX)
    %template(makeBackground) lsst::afw::math::makeBackground<lsst::afw::image::Image<PIXTYPE> >;
    %template(makeBackground) lsst::afw::math::makeBackground<lsst::afw::image::MaskedImage<PIXTYPE> >;
    %template(BackgroundMI ## SUFFIX) lsst::afw::math::BackgroundMI::BackgroundMI<lsst::afw::image::Image<PIXTYPE> >;
    %template(getImage ## SUFFIX) lsst::afw::math::Background::getImage<PIXTYPE>;
    %template(getImage ## SUFFIX) lsst::afw::math::BackgroundMI::getImage<PIXTYPE>;
%enddef

%declareBack(float, F)


