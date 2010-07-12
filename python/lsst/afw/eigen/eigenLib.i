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
 
%define eigenLib_DOCSTRING
"
Interface to interchange Eigen matrices and numpy arrays
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.eigen", docstring=eigenLib_DOCSTRING) eigenLib

%pythonnondynamic;
%naturalvar;  // use const reference typemaps

#define SWIG_FILE_WITH_INIT
%include "eigen.i"

/************************************************************************************************************/
/*
 * Helper macro which defines the desired matrix sizes, e.g. Dynamic*Dynamic
 */
%define %numpyToEigenAllDims(MACRO, NUMPY_T, C_T)
    MACRO(NUMPY_T, C_T, Eigen::Dynamic, Eigen::Dynamic);
%enddef
/*
 * Helper macro to generate typedefs for all desired matrix sizes
 */
%define %numpy_to_eigen(NUMPY_T, C_T)
    %eigenToNumpy(NUMPY_T, C_T, 2, 2);

    %numpyToEigenAllDims(%eigenToNumpy,    NUMPY_T, C_T);
    %numpyToEigenAllDims(%numpyToEigenPtr, NUMPY_T, C_T);
    %numpyToEigenAllDims(%numpyToEigenRef, NUMPY_T, C_T);
%enddef
/*
 * Actually generate the typedefs
 */
%numpy_to_eigen(PyArray_DOUBLE, double);
%numpy_to_eigen(PyArray_FLOAT, float);
%numpy_to_eigen(PyArray_INT, int);
%numpy_to_eigen(PyArray_LONG, long);
