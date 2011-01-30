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
 
//See numpyTypemaps.h for documentation and examples

%{
    #include "lsst/ndarray/python/numpy.hpp"
%}

%define %declareEigenMatrix(TYPE...)
%typemap(out) TYPE {
    numpy_typemaps::PyPtr tmp = numpy_typemaps::EigenPyConverter< TYPE >::toPython($1);
    $result = tmp.get();
    Py_XINCREF($result);
}
%typemap(out) TYPE const & {
    numpy_typemaps::PyPtr tmp = numpy_typemaps::EigenPyConverter< TYPE >::toPython(*$1);
    $result = tmp.get();
    Py_XINCREF($result);
}
%typemap(typecheck) TYPE, TYPE const *, TYPE const & {
    numpy_typemaps::PyPtr tmp($input,true);
    $1 = numpy_typemaps::EigenPyConverter< TYPE >::fromPythonStage1(tmp);
    if (!$1) PyErr_Clear();
}
%typemap(in) TYPE const & (TYPE val) {
    numpy_typemaps::PyPtr tmp($input,true);
    if (!numpy_typemaps::EigenPyConverter< TYPE >::fromPythonStage2(tmp, val)) return NULL;
    $1 = &val;
}
%enddef
