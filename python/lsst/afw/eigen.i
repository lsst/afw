%{
    #include "lsst/afw/numpyTypemaps.h"
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
