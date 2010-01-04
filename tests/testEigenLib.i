// -*- lsst-c++ -*-
%define testEigenLib_DOCSTRING
"
Various swigged-up C++ classes for testing
"
%enddef

%feature("autodoc", "1");
%module(package="testEigenLib", docstring=testEigenLib_DOCSTRING) testEigenLib

%pythonnondynamic;
%naturalvar;  // use const reference typemaps

%{
    #include <iostream>
    #include "boost/shared_ptr.hpp"
    #include <numpy/arrayobject.h>
    #include "Eigen/Core.h"
%}

%include "lsst/afw/eigen/eigenLib.i"

%feature("valuewrapper") Eigen::MatrixBase<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor > >;

%ignore Wcs;                            // The real C++ version; see also WcsWrapper

%inline %{
#define PRINT 0                         // print in test code
    
    void Wcs(Eigen::Matrix<double, 2, 2, Eigen::RowMajor> const& CD) {
#if PRINT
        std::cout << "I am CD" << std::endl;
#endif
    }
    
    template<typename Derived>
    void printIt(Eigen::MatrixBase<Derived> const& mat) {
#if PRINT
        std::cout << "printIt" << std::endl;

        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                std::cout << mat(i, j) << " ";
            }
            std::cout << std::endl;
        }
#endif
    }

    template<typename Derived>
    void identity(Eigen::MatrixBase<Derived> *mat) {
        for (int i = 0; i < mat->rows(); ++i) {
            for (int j = 0; j < mat->cols(); ++j) {
                (*mat)(i, j) = (i == j) ? 1.0 : 0.0;
            }
        }

        Eigen::Matrix<double, 2, 2, Eigen::RowMajor> cd;
        cd << 1, 2, 3, 4;

        printIt(cd);
        Wcs(cd);
    }

    template<int ND>
    Eigen::Matrix<double, ND, ND, Eigen::RowMajor> getIdentityN() {
        Eigen::Matrix<double, ND, ND, Eigen::RowMajor> mat;
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                mat(i, j) = (i == j) ? 1.0 : 0.0;
            }
        }

        return mat;
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> getIdentity(int const nd) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat(nd, nd);
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                mat(i, j) = (i == j) ? 1.0 : 0.0;
            }
        }

        return mat;
    }
%}

/*
 * Provide a swiggable wrapper to the Wcs function that uses the eigen/numpy typemaps
 */
%inline %{
    template<typename Derived>
    void WcsWrapper(Eigen::MatrixBase<Derived> const& CD) {
        if (CD.rows() != 2 || CD.cols() != 2) {
            SWIG_exception_fail(0, "CD Matrix must be 2x2");
        fail:
            return;
        }
        Wcs(CD);
    }
%}
%template(Wcs) WcsWrapper<Eigen::Map<%MatrixXnumpy(double, Eigen::Dynamic, Eigen::Dynamic)> >;

//
%define %declareTemplates(C_T)
    %template(getIdentity2) getIdentityN<2>;

    %template(identity) identity<Eigen::Map<%MatrixXnumpy(C_T, Eigen::Dynamic, Eigen::Dynamic)> >;
    %template(printIt)  printIt<Eigen::Map<%MatrixXnumpy(C_T, Eigen::Dynamic, Eigen::Dynamic)> >;
%enddef

%declareTemplates(double);
%declareTemplates(float);
%declareTemplates(int);
