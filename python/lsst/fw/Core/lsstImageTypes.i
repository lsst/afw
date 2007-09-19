// -*- lsst-c++ -*-
/*****************************************************************************/

%import  <vw/Core/FundamentalTypes.h>
%apply int {int32};
%apply int {vw::int32};
%apply int {boost::int32_t};
%apply int {vw::uint32};
%apply int {boost::uint32_t};
%apply int {vw::uint16};
%apply int {boost::uint16_t};
%apply int {vw::int16_t};
%apply int {boost::int16_t};
%apply char {vw::uint8};
%apply char {boost::uint8_t};

/******************************************************************************/
/*
 * Typemaps
 */

/*
 * Handle treating arrays within structures as arrays.
 *
 * N.b.
 *   struct.arr[0] = 123
 * won't work as expected
 */
%typemap(in) float [ANY] (float temp[$1_dim0]) {
    if (!PySequence_Check($input)) {
        PyErr_SetString(PyExc_ValueError,"Expected a sequence");
        return NULL;
    }
    if (PySequence_Length($input) != $1_dim0) {
        PyErr_SetString(PyExc_ValueError,"Size mismatch. Expected $1_dim0 elements");
        return NULL;
    }
    
    for (int i = 0; i < $1_dim0; i++) {
        PyObject *o = PySequence_GetItem($input,i);
        if (PyNumber_Check(o)) {
            temp[i] = (float) PyFloat_AsDouble(o);
        } else {
            PyErr_SetString(PyExc_ValueError,"Sequence elements must be numbers");      
            return NULL;
        }
    }

    $1 = temp;
}

%typemap(memberin) float [ANY] {
    for (int i = 0; i < $1_dim0; i++) {        // memberin
        $1[i] = $input[i];
    }
}

%typemap(memberin) double [ANY] {
    for (int i = 0; i < $1_dim0; i++) {        // memberin
        $1[i] = $input[i];
    }
}

%typemap(out) float [ANY] {
    $result = PyList_New($1_dim0);
    for (int i = 0; i < $1_dim0; i++) {
        PyObject *o = PyFloat_FromDouble((double) $1[i]);
        PyList_SetItem($result, i, o);
    }
}

/******************************************************************************/

%exception {
    try {
        $action
    } catch (lsst::mwi::exceptions::ExceptionStack &e) {
        PyErr_SetString(PyExc_IndexError, e.what());
        return NULL;
    } catch (vw::Exception &e) {
        PyErr_SetString(PyExc_IndexError, e.what());
        return NULL;
    }
}

