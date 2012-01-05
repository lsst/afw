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
 
%define tableLib_DOCSTRING
"
Python interface to lsst::afw::table classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.table", docstring=tableLib_DOCSTRING) tableLib

#pragma SWIG nowarn=389                 // operator[]  ignored
#pragma SWIG nowarn=503                 // comparison operators ignored

%{
#include "lsst/afw/table.h"
#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_TABLE_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "lsst/ndarray/python.h"
%}

%init %{
    import_array();
%}

%include "std_set.i"

%include "lsst/p_lsstSwig.i"

%lsst_exceptions();

%{
#include <sstream>
%}
%define %addStreamRepr(CLASS)
%extend CLASS {
    std::string __repr__() const {
        std::ostringstream os;
        os << (*self);
        return os.str();
    }
    std::string __str__() const {
        std::ostringstream os;
        os << (*self);
        return os.str();        
    }
}
%enddef

%include "lsst/ndarray/ndarray.i"

%include "lsst/afw/table/FieldBase.h"
%include "lsst/afw/table/Field.h"
%include "lsst/afw/table/KeyBase.h"
%include "lsst/afw/table/Key.h"
%include "lsst/afw/table/detail/SchemaImpl.h"

%rename("__eq__") lsst::afw::table::Schema::operator==;
%rename("__ne__") lsst::afw::table::Schema::operator!=;

%include "lsst/afw/table/Schema.h"

%pythoncode %{
Field = {}
Key = {}
SchemaItem = {}
_suffixes = {}
%}

%define %declareFieldType(CNAME, PYNAME)
%rename("_eq_impl") lsst::afw::table::Key< CNAME >::operator==;
%extend lsst::afw::table::Key< CNAME > {
    %pythoncode %{
         def __eq__(self, other):
             if type(other) != type(self): return NotImplemented
             return self._eq_impl(other)
         def __ne__(self, other): return not self == other
    %}
}
%template(FieldBase_ ## PYNAME) lsst::afw::table::FieldBase< CNAME >;
%template(Field_ ## PYNAME) lsst::afw::table::Field< CNAME >;
%template(KeyBase_ ## PYNAME) lsst::afw::table::KeyBase< CNAME >;
%template(Key_ ## PYNAME) lsst::afw::table::Key< CNAME >;
%template(SchemaItem_ ## PYNAME) lsst::afw::table::SchemaItem< CNAME >;
%addStreamRepr(lsst::afw::table::Field< CNAME >);
%addStreamRepr(lsst::afw::table::Key< CNAME >);
%pythoncode %{
Field[FieldBase_ ## PYNAME.getTypeString()] = Field_ ## PYNAME
Key[FieldBase_ ## PYNAME.getTypeString()] = Key_ ## PYNAME
SchemaItem[FieldBase_ ## PYNAME.getTypeString()] = SchemaItem_ ## PYNAME
_suffixes[FieldBase_ ## PYNAME.getTypeString()] = #PYNAME
%}
%extend lsst::afw::table::Schema {
    %template(_find_ ## PYNAME) find< CNAME >;
    %template(_addField_ ## PYNAME) addField< CNAME >;
    %template(replaceField) replaceField< CNAME >;
}
%implicitconv FieldBase_ ## PYNAME;
%enddef

%declareFieldType(boost::int32_t, I4)
%declareFieldType(boost::int64_t, I8)
%declareFieldType(float, F4)
%declareFieldType(double, F8)
%declareFieldType(lsst::afw::table::Flag, Flag)

%declareFieldType(lsst::afw::table::Point<boost::int32_t>, PointI4)
%declareFieldType(lsst::afw::table::Point<float>, PointF4)
%declareFieldType(lsst::afw::table::Point<double>, PointF8)

%declareFieldType(lsst::afw::table::Shape<float>, ShapeF4)
%declareFieldType(lsst::afw::table::Shape<double>, ShapeF8)

%declareFieldType(lsst::afw::table::Array<float>, ArrayF4)
%declareFieldType(lsst::afw::table::Array<double>, ArrayF8)

%declareFieldType(lsst::afw::table::Covariance<float>, CovF4)
%declareFieldType(lsst::afw::table::Covariance<double>, CovF8)

%declareFieldType(lsst::afw::table::Covariance< lsst::afw::table::Point<float> >, CovPointF4)
%declareFieldType(lsst::afw::table::Covariance< lsst::afw::table::Point<double> >, CovPointF8)

%declareFieldType(lsst::afw::table::Covariance< lsst::afw::table::Shape<float> >, CovShapeF4)
%declareFieldType(lsst::afw::table::Covariance< lsst::afw::table::Shape<double> >, CovShapeF8)

%include "bug3465431.i"

%template(NameSet) std::set<std::string>;

%extend lsst::afw::table::Schema {

%pythoncode %{

def find(self, k):
    if not isinstance(k, basestring):
         try:
             prefix, suffix = type(k).__name__.split("_")
         except Exception:
             raise TypeError("Argument to Schema.find must be a string or Key.")
         if prefix != "Key":
             raise TypeError("Argument to Schema.find must be a string or Key.")
         attr = "_find_" + suffix
         method = getattr(self, attr)
         return method(k)
    for suffix in _suffixes.itervalues():
         attr = "_find_" + suffix
         method = getattr(self, attr)
         try:
             return method(k)
         except Exception:
             pass
    raise KeyError("Field '%s' not found in Schema." % k)

def addField(self, field, type=None, doc="", units="", size=None):
    if type is None:
        try:
            prefix, suffix = __builtins__.type(field).__name__.split("_")
        except Exception:
            raise TypeError("First argument to Schema.find must be a Field if 'type' is not given.")
        if prefix != "Field":
            raise TypeError("First argument to Schema.find must be a Field if 'type' is not given.")
        attr = "_addField_" + suffix
        method = getattr(self, attr)
        return method(field)
    suffix = _suffixes[type]
    attr = "_addField_" + suffix
    method = getattr(self, attr)
    if size is None:
        size = globals()["FieldBase_" + suffix]()
    else:
        size = globals()["FieldBase_" + suffix](size)
    return method(field, doc, units, size)

%}


}
