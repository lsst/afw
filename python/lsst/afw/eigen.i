%{
#if 0
 } // unconfuse emacs
#endif

#include "Eigen/Core"
#include "boost/intrusive_ptr.hpp"

namespace boost {
inline void intrusive_ptr_add_ref(PyObject * obj) { Py_INCREF(obj); }
inline void intrusive_ptr_release(PyObject * obj) { Py_DECREF(obj); }
}

namespace numpy_typemaps {

/** 
 *  \internal \ingroup PythonInternalGroup
 *  \brief Traits class that specifies Numpy typecodes for numeric types. 
 */
template <typename T> struct NumpyTraits { 
    static int getCode(); 
};

/// \cond SPECIALIZATIONS

template <> struct NumpyTraits<bool> {
    static int getCode() {
	if (sizeof(bool)==sizeof(npy_bool)) return NPY_BOOL;
	if (sizeof(bool)==1) return NPY_UBYTE;
	if (sizeof(bool)==2 && sizeof(short)==2) return NPY_USHORT;
	if (sizeof(bool)==4 && sizeof(int)==4) return NPY_UINT;
	assert(false);
	return 0;
    }
};

template <> struct NumpyTraits<npy_ubyte> { static int getCode() { return NPY_UBYTE; } };
template <> struct NumpyTraits<npy_byte> { static int getCode() { return NPY_BYTE; } };
template <> struct NumpyTraits<npy_ushort> { static int getCode() { return NPY_USHORT; } };
template <> struct NumpyTraits<npy_short> { static int getCode() { return NPY_SHORT; } };
template <> struct NumpyTraits<npy_uint> { static int getCode() { return NPY_UINT; } };
template <> struct NumpyTraits<npy_int> { static int getCode() { return NPY_INT; } };
template <> struct NumpyTraits<npy_ulong> { static int getCode() { return NPY_ULONG; } };
template <> struct NumpyTraits<npy_long> { static int getCode() { return NPY_LONG; } };
template <> struct NumpyTraits<npy_ulonglong> { static int getCode() { return NPY_ULONGLONG; } };
template <> struct NumpyTraits<npy_longlong> { static int getCode() { return NPY_LONGLONG; } };
template <> struct NumpyTraits<npy_float> { static int getCode() { return NPY_FLOAT; } };
template <> struct NumpyTraits<npy_double> { static int getCode() { return NPY_DOUBLE; } };
template <> struct NumpyTraits<npy_longdouble> { static int getCode() { return NPY_LONGDOUBLE; } };
template <> struct NumpyTraits<npy_cfloat> { static int getCode() { return NPY_CFLOAT; } };
template <> struct NumpyTraits<npy_cdouble> { static int getCode() { return NPY_CDOUBLE; } };
template <> struct NumpyTraits<npy_clongdouble> { static int getCode() { return NPY_CLONGDOUBLE; } };

template <> struct NumpyTraits<std::complex<float> > { 
    static int getCode() { assert(sizeof(std::complex<float>)==sizeof(npy_cfloat)); return NPY_CFLOAT; } 
};

template <> struct NumpyTraits<std::complex<double> > { 
    static int getCode() { assert(sizeof(std::complex<double>)==sizeof(npy_cdouble)); return NPY_CDOUBLE; } 
};

template <> struct NumpyTraits<std::complex<long double> > { 
    static int getCode() { 
	assert(sizeof(std::complex<long double>)==sizeof(npy_clongdouble)); 
	return NPY_CLONGDOUBLE; 
    }
};

/**
 *  \brief A reference-counting smart pointer for PyObject.
 */
typedef boost::intrusive_ptr<PyObject> PyPtr;

template <typename Matrix>
class EigenPyConverter {
    typedef typename Matrix::Scalar Scalar;
    typedef boost::mpl::int_<Matrix::RowsAtCompileTime> Rows;
    typedef boost::mpl::int_<Matrix::ColsAtCompileTime> Cols;
    typedef boost::mpl::int_<Matrix::SizeAtCompileTime> Size;
    typedef Eigen::Matrix<Scalar,Rows::value,Cols::value> TrueMatrix;

    static PyPtr getNumpyMatrixType() {
        PyPtr numpyModule(PyImport_ImportModule("numpy"),false);
        if (numpyModule) {
            return PyPtr(PyObject_GetAttrString(numpyModule.get(),"matrix"),false);
        }
        return PyPtr();
    }

    static PyPtr makeNumpyMatrix(PyPtr const & array) {
        PyPtr matrixType(getNumpyMatrixType());
        if (!matrixType) return PyPtr();
        PyPtr args(PyTuple_Pack(1,array.get()),false);
        PyPtr kwds(PyDict_New(),false);
        if (PyDict_SetItemString(kwds.get(),"copy",Py_False) != 0) return PyPtr();
        return PyPtr(PyObject_Call(matrixType.get(),args.get(),kwds.get()),false);
    }

public:

    /** 
     *  \brief Convert a C++ object to a new Python object.
     *
     *  \return A new Python object, or NULL on failure (with
     *  a Python exception set).
     */
    static PyPtr toPython(
        Matrix const & input, ///< Input C++ object.
        bool squeeze = false
    ) {
        npy_intp shape[2] = { input.rows(), input.cols() };
        PyPtr array(PyArray_SimpleNew(2, shape, NumpyTraits<Scalar>::getCode()));
        for (int i=0; i<input.rows(); ++i) {
            for (int j=0; j<input.cols(); ++j) {
                *((Scalar*)PyArray_GETPTR2(array.get(), i, j)) = input(i,j);
            }
        }
        PyPtr r;
        if (squeeze) {
            r.reset(PyArray_Squeeze(reinterpret_cast<PyArrayObject*>(array.get())));
        } else {
            r = makeNumpyMatrix(array);
        }
        return r;
    }

    /**
     *  \brief Check if a Python object is convertible to T
     *  and optionally begin the conversion by replacing the
     *  input with an intermediate.
     *
     *  \return true if a conversion may be possible, and
     *  false if it is not (with a Python exception set).
     */
    static bool fromPythonStage1(
        PyPtr & p /**< On input, a Python object to be converted.
                   *   On output, a Python object to be passed to
                   *   fromPythonStage2().
                   */
    ) {
        PyPtr tmp = makeNumpyMatrix(p);
        if (!tmp) return false;
        if (Cols::value != Eigen::Dynamic && Cols::value != PyArray_DIM(tmp.get(),1)) {
            if (Cols::value == 1 && PyArray_DIM(tmp.get(),0) == 1) { 
                tmp = PyPtr(PyObject_CallMethod(tmp.get(),const_cast<char*>("transpose"),NULL),false);
            } else {
                PyErr_SetString(PyExc_ValueError,"Incorrect number of columns for matrix.");
                return false;
            }
        }
        if (Rows::value != Eigen::Dynamic && Rows::value != PyArray_DIM(tmp.get(),0)) {
            PyErr_SetString(PyExc_ValueError,"Incorrect number of rows for matrix.");
            return false;
        }
        p = tmp;
        return true;
    }

    /**
     *  \brief Complete a Python to C++ conversion begun
     *  with fromPythonStage1().
     *
     *  \return true if the conversion was successful,
     *  and false otherwise (with a Python exception set).
     */
    static bool fromPythonStage2(
        PyPtr const & p, ///< A Python object processed by fromPythonStage1().
        Matrix & output       ///< The output C++ object.
    ) {
        PyPtr array(p);
        if (!fromPythonStage1(array)) return false;
        output = Matrix::Zero(PyArray_DIM(array.get(), 0), PyArray_DIM(array.get(), 1));
        for (int i=0; i<output.rows(); ++i) {
            for (int j=0; j<output.cols(); ++j) {
                output(i,j) = *((Scalar*)PyArray_GETPTR2(array.get(), i, j));
            }
        }
        return true;
    }

};

} // namespace numpy_typemaps

#if 0
{ // unconfuse emacs
#endif
    %}

%define %declareEigenMatrix(TYPE)
%typemap(out) TYPE {
    numpy_typemaps::PyPtr tmp = numpy_typemaps::EigenPyConverter< TYPE >::toPython($1);
    $result = tmp.get();
    Py_XINCREF($result);
}
%typemap(out) TYPE const & {
    $result = numpy_typemaps::EigenPyConverter< TYPE >::toPython(*$1);
}
%typemap(typecheck) TYPE, TYPE const * {
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
