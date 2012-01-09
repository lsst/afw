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

%include "lsst/ndarray/ndarray.i"
%init %{
    import_array();
%}
%declareNumPyConverters(lsst::ndarray::Array<boost::int32_t const,1>);
%declareNumPyConverters(lsst::ndarray::Array<float const,1>);
%declareNumPyConverters(lsst::ndarray::Array<double const,1>);
%declareNumPyConverters(Eigen::Array<float,Eigen::Dynamic,1>);
%declareNumPyConverters(Eigen::Array<double,Eigen::Dynamic,1>);
%declareNumPyConverters(Eigen::Matrix<float,2,2>);
%declareNumPyConverters(Eigen::Matrix<double,2,2>);
%declareNumPyConverters(Eigen::Matrix<float,3,3>);
%declareNumPyConverters(Eigen::Matrix<double,3,3>);
%declareNumPyConverters(Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>);
%declareNumPyConverters(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>);

%include "std_set.i"

%include "lsst/p_lsstSwig.i"

%lsst_exceptions();

%import "lsst/afw/geom/geomLib.i"
%import "lsst/afw/geom/ellipses/ellipsesLib.i"

// ------------------ General purpose stuff that maybe should go in p_lsstSwig.i ---------------------------

%{
#include <sstream>
%}
%include "std_container.i"

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

%define %returnNone(FUNC)
%feature("pythonappend") FUNC %{ val = None %}
%enddef

%define %makeIterable(CLASS, VALUE)
%fragment("SwigPyIterator_T");
%fragment("StdTraits");
%newobject __iter__(PyObject **PYTHON_SELF);
%{
namespace swig {
template <> struct traits< VALUE > {
    typedef value_category category;
    static const char * type_name() { return # VALUE; }
};
} // namespace swig
%}
%extend CLASS {
    swig::SwigPyIterator * __iter__(PyObject** PYTHON_SELF) {
        return swig::make_output_iterator(self->begin(), self->begin(), self->end(), *PYTHON_SELF);
    }
}
%enddef

// ---------------------------------------------------------------------------------------------------------

%include "lsst/ndarray/ndarray.i"

%shared_ptr(lsst::afw::table::AuxBase);
%shared_ptr(lsst::afw::table::IdFactory);
%ignore lsst::afw::table::IdFactory::operator=;

%include "lsst/base.h"
%include "lsst/afw/table/misc.h"
%include "lsst/afw/table/ModificationFlags.h"
%include "lsst/afw/table/IdFactory.h"
%include "lsst/afw/table/FieldBase.h"
%include "lsst/afw/table/Field.h"
%include "lsst/afw/table/KeyBase.h"
%include "lsst/afw/table/Key.h"
%include "lsst/afw/table/detail/SchemaImpl.h"

%rename("__eq__") lsst::afw::table::Schema::operator==;
%rename("__ne__") lsst::afw::table::Schema::operator!=;

%include "lsst/afw/table/Schema.h"

%extend lsst::afw::table::Schema {

%pythoncode %{

def asList(self):
    # This should be replaced by an implementation that uses Schema::forEach
    # if/when SWIG gets better at handling templates or we switch to Boost.Python.
    result = []
    def extractSortKey(item):
        key = item.key
        if type(key) == Key_Flag:
            return (key.getOffset(), get.getBit())
        else:
            return (key.getOffset(), None)
    for name in self.getNames():
        result.append(self.find(name))
    result.sort(key=extractSortKey)
    return result

def __iter__(self):
    return iter(self.asList())

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
    if not isinstance(type, basestring):
        type = aliases[type]
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

%include "lsst/afw/table/SchemaMapper.h"

%ignore lsst::afw::table::RecordBase::operator=;
%rename("__eq__") lsst::afw::table::RecordBase::operator==;
%rename("__ne__") lsst::afw::table::RecordBase::operator!=;

%include "lsst/afw/table/RecordBase.h"

%include "lsst/afw/table/ColumnView.h"

%ignore lsst::afw::table::TableBase::begin;
%ignore lsst::afw::table::TableBase::end;
%ignore lsst::afw::table::TableBase::find;
%nodefaultctor lsst::afw::table::TableBase;
%rename(__getitem__) lsst::afw::tableTableBase::operator[];
%returnNone(lsst::afw::table::TableBase::unlink)
%makeIterable(lsst::afw::table::TableBase, lsst::afw::table::RecordBase)
%include "lsst/afw/table/TableBase.h"

%ignore lsst::afw::table::RecordInterface::operator<<=;
%ignore lsst::afw::table::RecordInterface::operator=;
%ignore lsst::afw::table::ChildView::begin;
%ignore lsst::afw::table::ChildView::end;
%ignore lsst::afw::table::TableInterface::begin;
%ignore lsst::afw::table::TableInterface::end;
%ignore lsst::afw::table::TableInterface::find;
%ignore lsst::afw::table::TableInterface::insert;
%nodefaultctor lsst::afw::table::TableInterface;
%rename(__getitem__) lsst::afw::table::TableInterface::operator[];
%returnNone(lsst::afw::table::TableInterface::unlink)

%include "lsst/afw/table/RecordInterface.h"
%include "lsst/afw/table/TableInterface.h"

%define %declareTag(TAG)
%template(TAG ## RecordInterface) lsst::afw::table::RecordInterface< lsst::afw::table::TAG >;
%template(TAG ## TableInterface) lsst::afw::table::TableInterface< lsst::afw::table::TAG >;
%template(TAG ## ChildView) lsst::afw::table::ChildView< lsst::afw::table::TAG >;
%enddef

%declareTag(Simple)
%include "lsst/afw/table/Simple.h"

%declareTag(Source)
%include "lsst/afw/table/Source.h"

%include "lsst/afw/table/ColumnView.h"

%pythoncode %{
import numpy
Field = {}
Key = {}
SchemaItem = {}
_suffixes = {}
aliases = {
    int: "I4",
    long: "I8",
    float: "F8",
    numpy.int32: "I4",
    numpy.int64: "I8",
    numpy.float32: "F4",
    numpy.float64: "F8",
}
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
%extend lsst::afw::table::SchemaMapper {
    %template(addOutputField) addOutputField< CNAME >;
    %template(addMapping) addMapping< CNAME >;
    %template(getMapping) getMapping< CNAME >;
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

%include "specializations.i"

%template(NameSet) std::set<std::string>;
