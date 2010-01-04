// -*- lsst-c++ -*-
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
