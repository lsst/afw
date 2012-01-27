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
#pragma SWIG nowarn=520                 // base class not similarly marked as smart pointer
#pragma SWIG nowarn=401                 // nothing known about base class

%lsst_exceptions();

%{
#include "lsst/afw/table.h"
#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_TABLE_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "lsst/ndarray/python.h"
#include "lsst/afw/geom/Angle.h"

// This enables numpy array conversion for Angle, converting it to a regular array of double.
namespace lsst { namespace ndarray { namespace detail {
template <> struct NumpyTraits<lsst::afw::geom::Angle> : public NumpyTraits<double> {};
}}}

%}

%include "lsst/ndarray/ndarray.i"
%init %{
    import_array();
%}

%declareNumPyConverters(lsst::ndarray::Array<bool const,1>);
%declareNumPyConverters(lsst::ndarray::Array<lsst::afw::table::RecordId const,1>);
%declareNumPyConverters(lsst::ndarray::Array<boost::int32_t const,1>);
%declareNumPyConverters(lsst::ndarray::Array<boost::int64_t const,1>);
%declareNumPyConverters(lsst::ndarray::Array<boost::int32_t,1,1>);
%declareNumPyConverters(lsst::ndarray::Array<boost::int64_t,1,1>);
%declareNumPyConverters(lsst::ndarray::Array<boost::int32_t const,1,1>);
%declareNumPyConverters(lsst::ndarray::Array<boost::int64_t const,1,1>);
%declareNumPyConverters(lsst::ndarray::Array<float const,1>);
%declareNumPyConverters(lsst::ndarray::Array<double const,1>);
%declareNumPyConverters(lsst::ndarray::Array<float,1,1>);
%declareNumPyConverters(lsst::ndarray::Array<double,1,1>);
%declareNumPyConverters(lsst::ndarray::Array<float const,1,1>);
%declareNumPyConverters(lsst::ndarray::Array<double const,1,1>);
%declareNumPyConverters(lsst::ndarray::Array<float const,2>);
%declareNumPyConverters(lsst::ndarray::Array<double const,2>);
%declareNumPyConverters(lsst::ndarray::Array<lsst::afw::geom::Angle const,1>);
%declareNumPyConverters(Eigen::Matrix<float,2,2>);
%declareNumPyConverters(Eigen::Matrix<double,2,2>);
%declareNumPyConverters(Eigen::Matrix<float,3,3>);
%declareNumPyConverters(Eigen::Matrix<double,3,3>);
%declareNumPyConverters(Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>);
%declareNumPyConverters(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>);

%include "lsst/p_lsstSwig.i"

%import "lsst/daf/base/baseLib.i"
%import "lsst/afw/geom/geomLib.i"
%import "lsst/afw/coord/coordLib.i"
%import "lsst/afw/geom/ellipses/ellipsesLib.i"

// We prefer to convert std::set<std::string> to a Python tuple, because SWIG's std::set wrapper
// doesn't do many of the things a we want it do (pretty printing, comparison operators, ...),
// and the expense of a deep copy shouldn't matter in this case.
%{
    inline PyObject * convertNameSet(std::set<std::string> const & input) {
        lsst::ndarray::PyPtr result(PyTuple_New(input.size()));
        if (!result) return 0;
        Py_ssize_t n = 0;
        for (std::set<std::string>::const_iterator i = input.begin(); i != input.end(); ++i, ++n) {
            PyObject * s = PyString_FromStringAndSize(i->data(), i->size());
            if (!s) return 0;
            PyTuple_SET_ITEM(result.get(), n, s);
        }
        Py_INCREF(result.get());
        return result.get();
    }
%}

%typemap(out) std::set<std::string> {
    $result = convertNameSet($1);
}

%typemap(out)
std::set<std::string> const &, std::set<std::string> &, std::set<std::string> const*, std::set<std::string>*
{
    $result = convertNameSet(*$1);
}

// ------------------ General purpose stuff for iteration over containers -----------------------------------

%include "std_container.i"

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

%shared_ptr(lsst::afw::table::BaseTable);
%shared_ptr(lsst::afw::table::BaseRecord);
%shared_ptr(lsst::afw::table::IdFactory);
%ignore lsst::afw::table::IdFactory::operator=;

%include "lsst/afw/table/misc.h"
%include "lsst/afw/table/IdFactory.h"
%include "lsst/afw/table/FieldBase.h"
%include "lsst/afw/table/Field.h"
%include "lsst/afw/table/KeyBase.h"
%include "lsst/afw/table/Key.h"
%include "lsst/afw/table/detail/SchemaImpl.h"

%rename("__eq__") lsst::afw::table::Schema::operator==;
%rename("__ne__") lsst::afw::table::Schema::operator!=;
%rename("__getitem__") lsst::afw::table::Schema::operator[];
%rename("__getitem__") lsst::afw::table::SubSchema::operator[];
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

} // %extend Schema

%extend lsst::afw::table::SubSchema {
%pythoncode %{
def find(self, k):
    for suffix in _suffixes.itervalues():
         attr = "_find_" + suffix
         method = getattr(self, attr)
         try:
             return method(k)
         except Exception:
             pass
    raise KeyError("Field '%s' not found in Schema." % self.getPrefix())    
def asField(self):
    for suffix in _suffixes.itervalues():
         attr = "_asField_" + suffix
         method = getattr(self, attr)
         try:
             return method()
         except Exception:
             pass
    raise KeyError("Field '%s' not found in Schema." % self.getPrefix())
def asKey(self):
    for suffix in _suffixes.itervalues():
         attr = "_asKey_" + suffix
         method = getattr(self, attr)
         try:
             return method()
         except Exception:
             pass
    raise KeyError("Field '%s' not found in Schema." % self.getPrefix())
%}
} // %extend SubSchema

%include "lsst/afw/table/SchemaMapper.h"

%include "lsst/afw/table/BaseTable.h"

%ignore lsst::afw::table::BaseRecord::operator=;
%rename("__eq__") lsst::afw::table::BaseRecord::operator==;
%rename("__ne__") lsst::afw::table::BaseRecord::operator!=;
%include "lsst/afw/table/BaseRecord.h"

%include "lsst/afw/table/ColumnView.h"

%pythoncode %{
from .. import geom
from .. import coord
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
    geom.Angle: "Angle",
    coord.Coord: "Coord",
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
%extend lsst::afw::table::SubSchema {
    %template(_find_ ## PYNAME) find< CNAME >;
    lsst::afw::table::Field< CNAME > _asField_ ## PYNAME() const { return *self; }
    lsst::afw::table::Key< CNAME > _asKey_ ## PYNAME() const { return *self; }
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
%declareFieldType(lsst::afw::geom::Angle, Angle)
%declareFieldType(lsst::afw::coord::Coord, Coord)

%declareFieldType(lsst::afw::table::Point<boost::int32_t>, PointI4)
%declareFieldType(lsst::afw::table::Point<float>, PointF4)
%declareFieldType(lsst::afw::table::Point<double>, PointF8)

%declareFieldType(lsst::afw::table::Moments<float>, MomentsF4)
%declareFieldType(lsst::afw::table::Moments<double>, MomentsF8)

%declareFieldType(lsst::afw::table::Array<float>, ArrayF4)
%declareFieldType(lsst::afw::table::Array<double>, ArrayF8)

%declareFieldType(lsst::afw::table::Covariance<float>, CovF4)
%declareFieldType(lsst::afw::table::Covariance<double>, CovF8)

%declareFieldType(lsst::afw::table::Covariance< lsst::afw::table::Point<float> >, CovPointF4)
%declareFieldType(lsst::afw::table::Covariance< lsst::afw::table::Point<double> >, CovPointF8)

%declareFieldType(lsst::afw::table::Covariance< lsst::afw::table::Moments<float> >, CovMomentsF4)
%declareFieldType(lsst::afw::table::Covariance< lsst::afw::table::Moments<double> >, CovMomentsF8)

%include "specializations.i"

%pythoncode %{
for d in (Field, Key, SchemaItem, _suffixes):
    for k, v in aliases.iteritems():
        d[k] = d[v]
%}

%shared_ptr(lsst::afw::table::SourceTable)
%shared_ptr(lsst::afw::table::SourceRecord)
// Workarounds for SWIG's failure to parse the Measurement template correctly.
// Otherwise we'd have one place in the code that controls all the canonical measurement types.
namespace lsst { namespace afw { namespace table {
     struct Flux {
         typedef Key< double > MeasKey;
         typedef Key< double > ErrKey;
         typedef double MeasValue;
         typedef double ErrValue;
     };
     struct Centroid {
         typedef Key< Point<double> > MeasKey;
         typedef Key< Covariance< Point<double> > > ErrKey;
         typedef lsst::afw::geom::Point<double,2> MeasValue;
         typedef Eigen::Matrix<double,2,2> ErrValue;
     };
     struct Shape {
         typedef Key< Moments<double> > MeasKey;
         typedef Key< Covariance< Moments<double> > > ErrKey;
         typedef lsst::afw::geom::ellipses::Quadrupole MeasValue;
         typedef Eigen::Matrix<double,3,3> ErrValue;
     };
}}}

%include "lsst/afw/table/Source.h"

%include "containers.i"
