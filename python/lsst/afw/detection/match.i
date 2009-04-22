// -*- lsst-c++ -*-

%{
#include "lsst/afw/detection/Source.h"
#include "lsst/afw/detection/SourceMatch.h"
%}

%ignore lsst::afw::detection::SourceMatch;

%typemap(out) std::vector<lsst::afw::detection::SourceMatch> {
    int len = (&$1)->size();
    $result = PyList_New(len);
    for (int i = 0; i < len; i++) {
        PyObject *tuple = PyTuple_New(3);
        if (tuple == 0) {
            SWIG_exception_fail(SWIG_MemoryError, "failed to allocate Python tuple");
        }
        boost::shared_ptr<lsst::afw::detection::Source> *source1 =
            new boost::shared_ptr<lsst::afw::detection::Source>((&$1)->operator[](i).get<0>());
        boost::shared_ptr<lsst::afw::detection::Source> *source2 =
            new boost::shared_ptr<lsst::afw::detection::Source>((&$1)->operator[](i).get<1>());
        PyObject *pySource1 = SWIG_NewPointerObj(SWIG_as_voidptr(source1),
            SWIGTYPE_p_boost__shared_ptrT_lsst__afw__detection__Source_t,
            SWIG_POINTER_OWN);
        PyObject *pySource2 = SWIG_NewPointerObj(SWIG_as_voidptr(source2),
            SWIGTYPE_p_boost__shared_ptrT_lsst__afw__detection__Source_t,
            SWIG_POINTER_OWN);
        PyTuple_SetItem(tuple, 0, pySource1);
        PyTuple_SetItem(tuple, 1, pySource2);
        PyTuple_SetItem(tuple, 2, PyFloat_FromDouble((&$1)->operator[](i).get<2>()));
        PyList_SetItem($result, i, tuple);
    }
}

%include "lsst/afw/detection/SourceMatch.h"

