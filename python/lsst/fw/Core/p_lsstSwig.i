// -*- lsst-c++ -*-
#if defined(SWIGPYTHON)
/*
 * don't allow user to add attributes to C-defined classes; catches a variety of
 * typos when the user _thinks_ that they're setting a C-level object, but in
 * reality they're adding a new (and irrelevent) member to the class
 */
%pythonnondynamic;
#endif

%naturalvar;                            // use const reference typemaps

%include "cpointer.i"
%include "exception.i"
%include "std_list.i"
%include "std_string.i"
%include "std_vector.i"
%include "std_iostream.i"
%include "typemaps.i"

%include "carrays.i"

// N.b. these may interfere with the use of e.g. std_list.i for primitive types;
// you'll have to say e.g.
// %clear int &;
// %template(listInt)  std::list<int>;

%apply int &OUTPUT { int & };
%apply float &OUTPUT { float & };
%apply double &OUTPUT { double & };

%array_class(float, floatArray);
%array_class(double, doubleArray);

/******************************************************************************/

%{
#   include "lsst/fw/Exception.h"
%}

/******************************************************************************/
/*
 * Don't expose the entire boost::shared_ptr to swig; it is complicated...
 */
namespace boost {
    template<class T>
    class shared_ptr {
    public:
        shared_ptr(T *);
        ~shared_ptr();
        T *operator->() const;
        int use_count() const;
    };
}

/******************************************************************************/
/*
 * Typemaps
 */
%typemap(in) FILE * {
    if ($input == Py_None) {
	$1 = NULL;
    } else if (!PyFile_Check($input)) {
	PyErr_SetString(PyExc_TypeError, "Need a file!");
	goto fail;
    } else {
	$1 = PyFile_AsFile($input);
    }
}

%typemap(in) char * {
    if ($input == Py_None) {
	$1 = NULL;
    } else if (!PyString_Check($input)) {
	PyErr_SetString(PyExc_TypeError, "Need a string!");
	goto fail;
    } else {
	$1 = PyString_AsString($input);
    }
}

%typemap(freearg) char * {}

/******************************************************************************/
/*
 * Accept a python list/tuple of TYPE if the routine wants an array
 */
#if 0
%define ACCEPT_LIST_AS_ARRAY(TYPE)
%typemap(arginit) (const TYPE **) {
  $1 = NULL;
}
%typemap(in) (const TYPE **) {
    /* Check if is a list */
    $1 = NULL;
    if ($input == Py_None) {
	$1 = NULL;
    } else if (PyList_Check($input) || PyTuple_Check($input)) {
        int input_is_list = PyList_Check($input);
        int n = (input_is_list) ? PyList_Size($input) : PyTuple_Size($input);
	if (n > 0) {
	    $1 = psAlloc(n*sizeof(void *));
	
	    for (int i = 0; i < n; i++) {
		PyObject *o =  (input_is_list) ? PyList_GetItem($input,i) : PyTuple_GetItem($input,i);
	    
                if ((SWIG_ConvertPtr(o, (void **)&$1[i], $*1_descriptor,SWIG_POINTER_EXCEPTION)) == -1) {
                   PyErr_SetString(PyExc_TypeError,"Failed to convert element of array");
                   psFree($1);
                   return NULL;
                }
	    }
	}
    } else {
	PyErr_SetString(PyExc_TypeError,"not a list or tuple");
	return NULL;
    }
}

%typemap(freearg) (const TYPE **) {
    psFree($1);
}
%enddef

%define ACCEPT_LIST_AS_ARRAY_WITH_DIMEN(TYPE)
%typemap(arginit) (const TYPE **, const int) {
  $1 = NULL;
}
%typemap(in) (const TYPE **, const int) {
    /* Check if is a list */
    $1 = NULL;
    if ($input == Py_None) {
	$1 = NULL; $2 = 0;
    } else if (PyList_Check($input) || PyTuple_Check($input)) {
        int input_is_list = PyList_Check($input);
        $2 = (input_is_list) ? PyList_Size($input) : PyTuple_Size($input);
	if ($2 > 0) {
	    $1 = psAlloc($2*sizeof(void *));
	
	    for (int i = 0; i < $2; i++) {
		PyObject *o =  (input_is_list) ? PyList_GetItem($input,i) : PyTuple_GetItem($input,i);
	    
                if ((SWIG_ConvertPtr(o, (void **)&$1[i], $*1_descriptor,SWIG_POINTER_EXCEPTION)) == -1) {
                   PyErr_SetString(PyExc_TypeError,"Failed to convert element of array");
                   psFree($1);
                   return NULL;
                }
	    }
	}
    } else {
	PyErr_SetString(PyExc_TypeError,"not a list or tuple");
	return NULL;
    }
}

%typemap(freearg) (const TYPE **, const int) {
    psFree($1);
}
%enddef
#endif

/******************************************************************************/

/******************************************************************************/
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
    for (int i = 0; i < $1_dim0; i++) {	// memberin
	$1[i] = $input[i];
    }
}

%typemap(memberin) double [ANY] {
    for (int i = 0; i < $1_dim0; i++) {	// memberin
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
    } catch (lsst::Exception &e) {
        PyErr_SetString(PyExc_IndexError, e.what());
        return NULL;
    } catch (vw::Exception &e) {
        PyErr_SetString(PyExc_IndexError, e.what());
        return NULL;
    }
}

/******************************************************************************/
/*
 * Throw an exception if func returns NULL
 */
%define NOTNULL(func)
    %exception func {
        $action;
	if (result == NULL) {
	    $cleanup;
	}
    }
%enddef

/*
 * Throw an exception if func returns a negative value
 */
%define NOTNEGATIVE(func)
    %exception func {
        $action;
	if (result < 0) {
	    $cleanup;
	}
    }
%enddef

/******************************************************************************/

%define CAST(TYPE)
    %pointer_cast(void *, TYPE *, cast_ ## TYPE ## Ptr); // convert void pointer to (TYPE *)
%enddef

/******************************************************************************/
/*
 * Accept a python list of ints if the routine wants an array
 */
#if 0
%typemap(arginit) (const int *intArr, int nel) {
  $1 = NULL;
}
%typemap(in) (const int *intArr, int nel) {
    /* Check if is a list */
    $1 = NULL;
    if ($input == Py_None) {
	$1 = NULL; $2 = 0;
    } else if (PyList_Check($input) || PyTuple_Check($input)) {
        int input_is_list = PyList_Check($input);
        $2 = (input_is_list) ? PyList_Size($input) : PyTuple_Size($input);
	if ($2 > 0) {
	    $1 = psAlloc($2*sizeof(int));
	
	    for (int i = 0; i < $2; i++) {
		PyObject *o =  (input_is_list) ? PyList_GetItem($input,i) : PyTuple_GetItem($input,i);
                if (PyInt_Check(o)) {
		    $1[i] = PyInt_AsLong((input_is_list) ?
                                         PyList_GetItem($input, i) : PyTuple_GetItem($input, i));
		} else {
		    PyErr_SetString(PyExc_TypeError, "list/tuple must contain numbers");
		    psFree($1);
		    return NULL;
		}
	    }
	}
    } else {
	PyErr_SetString(PyExc_TypeError,"not a list or tuple");
	return NULL;
    }
}

%typemap(freearg) (const int *intArr, int nel) {
    psFree($1);
}

%typemap(arginit) (const int *) {
  $1 = NULL;
}
%typemap(in) (const int *) {
    /* Check if is a list */
    $1 = NULL;
    if ($input == Py_None) {
	$1 = NULL;
    } else if (PyList_Check($input) || PyTuple_Check($input)) {
        int input_is_list = PyList_Check($input);
        int n = (input_is_list) ? PyList_Size($input) : PyTuple_Size($input);
	if (n > 0) {
	    $1 = psAlloc(n*sizeof(int));
	
	    for (int i = 0; i < n; i++) {
		PyObject *o =  (input_is_list) ? PyList_GetItem($input,i) : PyTuple_GetItem($input,i);
                if (PyInt_Check(o)) {
		    $1[i] = PyInt_AsLong((input_is_list) ?
                                         PyList_GetItem($input, i) : PyTuple_GetItem($input, i));
		} else {
		    PyErr_SetString(PyExc_TypeError, "list/tuple must contain numbers");
		    psFree($1);
		    return NULL;
		}
	    }
	}
    } else {
	PyErr_SetString(PyExc_TypeError,"not a list or tuple");
	return NULL;
    }
}

%typemap(freearg) (const int *) {
    psFree($1);
}

/******************************************************************************/
/*
 * Accept a python list of floats if the routine wants an array
 */
%typemap(arginit) (const float *floatArr, int nel) {
  $1 = NULL;
}
%typemap(in) (const float *floatArr, int nel) {
    /* Check if is a list */
    $1 = NULL;
    if ($input == Py_None) {
	$1 = NULL; $2 = 0;
    } else if (PyList_Check($input) || PyTuple_Check($input)) {
        int input_is_list = PyList_Check($input);
        $2 = (input_is_list) ? PyList_Size($input) : PyTuple_Size($input);
	if ($2 > 0) {
	    $1 = psAlloc($2*sizeof(float));
	
	    for (int i = 0; i < $2; i++) {
		PyObject *o =  (input_is_list) ? PyList_GetItem($input,i) : PyTuple_GetItem($input,i);
		if (PyFloat_Check(o)) {
		    $1[i] = PyFloat_AsDouble((input_is_list) ?
                                             PyList_GetItem($input, i) : PyTuple_GetItem($input, i));
                } else if (PyInt_Check(o)) {
		    $1[i] = PyInt_AsLong((input_is_list) ?
                                         PyList_GetItem($input, i) : PyTuple_GetItem($input, i));
                } else {
		    PyErr_SetString(PyExc_TypeError, "list/tuple must contain numbers");
		    psFree($1);
		    return NULL;
		}
	    }
	}
    } else {
	PyErr_SetString(PyExc_TypeError,"not a list or tuple");
	return NULL;
    }
}

%typemap(freearg) (const float *floatArr, int nel) {
    psFree($1);
}

/******************************************************************************/
/*
 * Accept a python list of doubles if the routine wants an array
 */
%typemap(arginit) (const double *doubleArr, int nel) {
  $1 = NULL;
}
%typemap(in) (const double *doubleArr, int nel) {
    /* Check if is a list */
    $1 = NULL;
    if ($input == Py_None) {
	$1 = NULL; $2 = 0;
    } else if (PyList_Check($input) || PyTuple_Check($input)) {
        int input_is_list = PyList_Check($input);
        $2 = (input_is_list) ? PyList_Size($input) : PyTuple_Size($input);
	if ($2 > 0) {
	    $1 = psAlloc($2*sizeof(double));
	
	    for (int i = 0; i < $2; i++) {
		PyObject *o =  (input_is_list) ? PyList_GetItem($input,i) : PyTuple_GetItem($input,i);
		if (PyFloat_Check(o)) {
		    $1[i] = PyFloat_AsDouble((input_is_list) ?
                                             PyList_GetItem($input, i) : PyTuple_GetItem($input, i));
                } else if (PyInt_Check(o)) {
		    $1[i] = PyInt_AsLong((input_is_list) ?
                                         PyList_GetItem($input, i) : PyTuple_GetItem($input, i));
		} else {
		    PyErr_SetString(PyExc_TypeError, "list/tuple must contain numbers");
		    psFree($1);
		    return NULL;
		}
	    }
	}
    } else {
	PyErr_SetString(PyExc_TypeError,"not a list or tuple");
	return NULL;
    }
}

%typemap(freearg) (const double *doubleArr, int nel) {
    psFree($1);
}
#endif

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
