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
 
/*
** Based on the work by Ricard Marxer in rikrd-loudia; heavily modified but still
** recognisable and thus his copyright still applies
**
** Copyright (C) 2008, 2009 Ricard Marxer <email@ricardmarxer.com>
**                                                                  
** This program is free software; you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation; either version 3 of the License, or   
** (at your option) any later version.                                 
**                                                                     
** This program is distributed in the hope that it will be useful,     
** but WITHOUT ANY WARRANTY; without even the implied warranty of      
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the       
** GNU General Public License for more details.                        
**                                                                     
** You should have received a copy of the GNU General Public License   
** along with this program; if not, write to the Free Software         
** Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA.
*/

%{
    #define PY_ARRAY_UNIQUE_SYMBOL lsst_afw_numpy

    #include "boost/shared_ptr.hpp"
%}

#if defined(SWIG_FILE_WITH_INIT)        // defined in swig
%{
    #define SWIG_FILE_WITH_INIT         // defined in C
%}
%init %{
    import_array();
%}
#endif

%include "numpy.i"

//
// Define an Eigen matrix corresponding to the desired numpy array of the given C type
//
%define %MatrixXnumpy(C_T, COL_DIMEN, ROW_DIMEN)
Eigen::Matrix<C_T, COL_DIMEN, ROW_DIMEN, Eigen::RowMajor>
%enddef

//
// Typechecks for a given C modifier (e.g. const& )
//
%define %eigen_typecheck(PRECEDENCE, NUMPY_T, C_T, TYPE_MODIFIER, COL_DIMEN, ROW_DIMEN)
%typecheck(PRECEDENCE, fragment="NumPy_Fragments") 
    (Eigen::MatrixBase<Eigen::Map<%MatrixXnumpy(C_T, COL_DIMEN, ROW_DIMEN)> > TYPE_MODIFIER) {
    $1 = is_array($input) && array_type($input) == NUMPY_T;
}
%enddef
/*
 * Helper macro to define needed typechecks given a modifier (e.g. const&) and matrix dimensions
 */
%define %eigen_typechecks(TYPE_MODIFIER, COL_DIMEN, ROW_DIMEN)
    %eigen_typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY, PyArray_DOUBLE, double, TYPE_MODIFIER, COL_DIMEN, ROW_DIMEN);
    %eigen_typecheck(SWIG_TYPECHECK_FLOAT_ARRAY,  PyArray_FLOAT,  float,  TYPE_MODIFIER, COL_DIMEN, ROW_DIMEN);
    %eigen_typecheck(SWIG_TYPECHECK_INT32_ARRAY,  PyArray_INT,    int,    TYPE_MODIFIER, COL_DIMEN, ROW_DIMEN);
%enddef

/************************************************************************************************************/
//
// Now for two all-to-similar typemaps, for "const&", and "Matrix *".  We don't support "const Matrix" due to
// problems with swig copy constructors, and the firm words about passing by value in the Eigen manual
//
// Typemaps for a Matrix const&; no need to ensure that we do no copying of the Matrix::Map here
// so we'll accept any type that numpy's willing to convert (n.b. int32 isn't convertable to float)
//
%define %numpyToEigenRef(NUMPY_T, C_T, COL_DIMEN, ROW_DIMEN)

%eigen_typechecks(const&, COL_DIMEN, ROW_DIMEN);

%typemap(in,
         fragment="NumPy_Fragments") 
Eigen::MatrixBase<Eigen::Map<%MatrixXnumpy(C_T, COL_DIMEN, ROW_DIMEN)> > const&
         (boost::shared_ptr<Eigen::Map<%MatrixXnumpy(C_T, COL_DIMEN, ROW_DIMEN)> > temp) {
    // Note that we use a boost::shared_ptr as temp must be at function scope, and assigning
    // Eigen::Maps doesn't copy the underlying pointer, so we need to create *temp at the
    // scope of this typemap.

    // create array from input
    int newObject = 0;
    PyArrayObject *in_array =
        obj_to_array_contiguous_allow_conversion($input, array_type($input), &newObject);

    if (in_array == NULL) {
        return NULL;
    }
    
    // require one or two dimensions
    int dims[] = {1, 2};
    require_dimensions_n(in_array, dims, 2);
    
    // get the dimensions
    int in_rows;
    int in_cols;
    if (array_numdims(in_array) == 2) {
        in_rows = array_size(in_array, 0);
        in_cols = array_size(in_array, 1);
    } else {
        in_rows = 1;
        in_cols = array_size(in_array, 0);
    }
    
    //
    // Build an Eigen::Map around the in_array's data
    //
    switch( array_type($input) ) {
      case PyArray_LONG:
      case PyArray_DOUBLE:
        {
            %MatrixXnumpy(C_T, COL_DIMEN, ROW_DIMEN) temp2$argnum =
                Eigen::Map<%MatrixXnumpy(double, COL_DIMEN, ROW_DIMEN)>((double *)array_data(in_array),
                                                                        in_rows, in_cols).cast<C_T>();

            temp.reset(new Eigen::Map<%MatrixXnumpy(C_T, COL_DIMEN, ROW_DIMEN)>(temp2$argnum.data(),
                                                          temp2$argnum.cols(), temp2$argnum.rows()));
        }
        break;

      case PyArray_INT:
        {
            %MatrixXnumpy(C_T, COL_DIMEN, ROW_DIMEN) temp2$argnum =
                Eigen::Map<%MatrixXnumpy(int, COL_DIMEN, ROW_DIMEN)>((int *)array_data(in_array),
                                                                     in_rows, in_cols).cast<C_T>();

            temp.reset(new Eigen::Map<%MatrixXnumpy(C_T, COL_DIMEN, ROW_DIMEN)>(temp2$argnum.data(),
                                                          temp2$argnum.cols(), temp2$argnum.rows()));
        }
        break;

      case PyArray_FLOAT:
        {
            %MatrixXnumpy(C_T, COL_DIMEN, ROW_DIMEN) temp2$argnum =
                Eigen::Map<%MatrixXnumpy(float, COL_DIMEN, ROW_DIMEN)>((float *)array_data(in_array),
                                                                       in_rows, in_cols).cast<C_T>();

            temp.reset(new Eigen::Map<%MatrixXnumpy(C_T, COL_DIMEN, ROW_DIMEN)>(temp2$argnum.data(),
                                                          temp2$argnum.cols(), temp2$argnum.rows()));
        }
        break;
        
      default:
        abort();                        // You can't get here; we checked array_type already
        return NULL;
    }

    $1 = temp.get();
}
%enddef

/************************************************************************************************************/
//
// Define a typemap to accept a numpy array of the specified type (both the numpy and C names)
// and call a function expecting an Eigen array. E.g.
//  %numpyToEigenPtr(PyArray_FLOAT, float, Eigen::Dynamic, Eigen::Dynamic);
//
// We use Eigen::Map<> to share memory between the numpy array and the Eigen array; this means that the Eigen
// code must be templated (so we can pass Eigen::Map<Eigen::Matrix> or Eigen::Matrix); the safe way to do this
// is with code such as
//
//    template<typename Derived>
//    void identity(Eigen::MatrixBase<Derived> *mat) {
//        std::cout << mat->cols() << " " << mat->rows << std::endl;
//    }
// where the Eigen::MatrixBase provides type safety
//
%define %numpyToEigenPtr(NUMPY_T, C_T, COL_DIMEN, ROW_DIMEN)
%eigen_typechecks(*, COL_DIMEN, ROW_DIMEN);

%typemap(in,
         fragment="NumPy_Fragments") 
         Eigen::MatrixBase<Eigen::Map<%MatrixXnumpy(C_T, COL_DIMEN, ROW_DIMEN)> > *
         (boost::shared_ptr<Eigen::Map<%MatrixXnumpy(C_T, COL_DIMEN, ROW_DIMEN)> > temp) {
    // Note that we use a boost::shared_ptr as temp must be at function scope, and assigning
    // Eigen::Maps doesn't copy the underlying pointer, so we need to create *temp at the
    // scope of this typemap.

    if (array_type($input) != NUMPY_T) {
        PyErr_SetString(PyExc_ValueError, "array must be of type C_T");
	return NULL;
    }

    // Extract C++ array from input
    PyArrayObject * in_array;
    int newObject = 0;
    in_array = obj_to_array_contiguous_allow_conversion($input, NUMPY_T, &newObject);

    if (in_array == NULL) {
        return NULL;
    }
    
    // require one or two dimensions
    int dims[] = {1, 2};
    require_dimensions_n(in_array, dims, 2);

    // get the dimensions
    int in_rows;
    int in_cols;
    if(array_numdims(in_array) == 2) {
        in_rows = array_size(in_array, 0);
        in_cols = array_size(in_array, 1);
    } else {
        in_rows = 1;
        in_cols = array_size(in_array, 0);
    }
    //
    // Build an Eigen::Map around the in_array's data
    //
    temp.reset(new Eigen::Map<%MatrixXnumpy(C_T, COL_DIMEN, ROW_DIMEN)>((C_T*)array_data(in_array),
                                                                        in_rows, in_cols));

    $1 = temp.get();
}
%enddef
 
/************************************************************************************************************/
/*
 * Return an Eigen::Matrix by value; both row-major and column-major are supported
 */
%define %eigenToNumpy(NUMPY_T, C_T, COL_DIMEN, ROW_DIMEN)
%typemap(out,
         fragment="NumPy_Fragments") %MatrixXnumpy(C_T, COL_DIMEN, ROW_DIMEN) {

    int const nd = 2;
    npy_intp dims[nd];
    dims[0] = $1.rows();
    dims[1] = $1.cols();
    $result = PyArray_SimpleNew(nd, dims, NUMPY_T);
    //
    // Set numpy array from the Eigen matrix
    //
    C_T *ptr = (C_T *)PyArray_DATA($result);
    for (int r = 0, i = 0; r != dims[0]; ++r) {
        for (int c = 0; c != dims[1]; ++c, ++i) {
            ptr[i] = $1(c, r);
        }
    }
}

%apply(%MatrixXnumpy(C_T, COL_DIMEN, ROW_DIMEN)) { Eigen::Matrix<C_T, COL_DIMEN, ROW_DIMEN> };

%enddef
