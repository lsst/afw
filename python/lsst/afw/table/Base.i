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
 * Wrappers for BaseTable, BaseRecord, BaseColumnView, and their dependencies (including Schema
 * and SchemaMapper and their components).  Also includes Catalog.i and instantiates BaseCatalog.
 *
 * This file does not include Simple-, Source-, or Exposure- Record/Table/Catalog, or the matching
 * functions.
 */

%{
#include "lsst/pex/logging.h"
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/coord.h"
#include "lsst/afw/fits.h"

#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/SchemaMapper.h"
#include "lsst/afw/table/BaseColumnView.h"
#include "lsst/afw/table/Catalog.h"

// This enables numpy array conversion for Angle, converting it to a regular array of double.
namespace ndarray { namespace detail {
template <> struct NumpyTraits<lsst::afw::geom::Angle> : public NumpyTraits<double> {};
}}

%}

// Macro that provides a Python-side dynamic cast.
// The BASE argument should be the root of the class hierarchy, not the immediate base class.
%define %addCastMethod(CLS, BASE)
%extend CLS {
    static PTR(CLS) _cast(PTR(BASE) base) {
        return boost::dynamic_pointer_cast< CLS >(base);
    }
}
%enddef

%include "ndarray.i"
%init %{
    import_array();
%}

%declareNumPyConverters(ndarray::Array<bool const,1>);
%declareNumPyConverters(ndarray::Array<lsst::afw::table::RecordId const,1>);
%declareNumPyConverters(ndarray::Array<boost::uint16_t const,1>);
%declareNumPyConverters(ndarray::Array<boost::int32_t const,1>);
%declareNumPyConverters(ndarray::Array<boost::int64_t const,1>);
%declareNumPyConverters(ndarray::Array<boost::uint16_t,1>);
%declareNumPyConverters(ndarray::Array<boost::int32_t,1>);
%declareNumPyConverters(ndarray::Array<boost::int64_t,1>);
%declareNumPyConverters(ndarray::Array<boost::uint16_t,1,1>);
%declareNumPyConverters(ndarray::Array<boost::int32_t,1,1>);
%declareNumPyConverters(ndarray::Array<boost::int64_t,1,1>);
%declareNumPyConverters(ndarray::Array<boost::uint16_t const,1,1>);
%declareNumPyConverters(ndarray::Array<boost::int32_t const,1,1>);
%declareNumPyConverters(ndarray::Array<boost::int64_t const,1,1>);
%declareNumPyConverters(ndarray::Array<int,1>);
%declareNumPyConverters(ndarray::Array<float,1>);
%declareNumPyConverters(ndarray::Array<double,1>);
%declareNumPyConverters(ndarray::Array<int const,1>);
%declareNumPyConverters(ndarray::Array<float const,1>);
%declareNumPyConverters(ndarray::Array<double const,1>);
%declareNumPyConverters(ndarray::Array<int,1,1>);
%declareNumPyConverters(ndarray::Array<float,1,1>);
%declareNumPyConverters(ndarray::Array<double,1,1>);
%declareNumPyConverters(ndarray::Array<int const,1,1>);
%declareNumPyConverters(ndarray::Array<float const,1,1>);
%declareNumPyConverters(ndarray::Array<double const,1,1>);
%declareNumPyConverters(ndarray::Array<boost::uint16_t,2>);
%declareNumPyConverters(ndarray::Array<int,2>);
%declareNumPyConverters(ndarray::Array<float,2>);
%declareNumPyConverters(ndarray::Array<double,2>);
%declareNumPyConverters(ndarray::Array<boost::uint16_t const,2>);
%declareNumPyConverters(ndarray::Array<int const,2>);
%declareNumPyConverters(ndarray::Array<float const,2>);
%declareNumPyConverters(ndarray::Array<double const,2>);
%declareNumPyConverters(ndarray::Array<lsst::afw::geom::Angle,1>);
%declareNumPyConverters(ndarray::Array<lsst::afw::geom::Angle const,1>);
%declareNumPyConverters(ndarray::Array<lsst::afw::table::BitsColumn::IntT,1,1>);
%declareNumPyConverters(ndarray::Array<lsst::afw::table::BitsColumn::IntT const,1,1>);
%declareNumPyConverters(Eigen::Matrix<float,2,2>);
%declareNumPyConverters(Eigen::Matrix<double,2,2>);
%declareNumPyConverters(Eigen::Matrix<float,3,3>);
%declareNumPyConverters(Eigen::Matrix<double,3,3>);
%declareNumPyConverters(Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>);
%declareNumPyConverters(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>);

%import "lsst/daf/base/baseLib.i"
%import "lsst/afw/geom/geomLib.i"
%import "lsst/afw/fits/fitsLib.i"
%import "lsst/afw/coord/coordLib.i"
%import "lsst/afw/geom/ellipses/ellipsesLib.i"

// =============== miscellaneous bits =======================================================================

%pythoncode %{
from . import _syntax
%}

%include "lsst/afw/table/misc.h"

// ---------------------------------------------------------------------------------------------------------

// We prefer to convert std::set<std::string> to a Python tuple, because SWIG's std::set wrapper
// doesn't do many of the things a we want it do (pretty printing, comparison operators, ...),
// and the expense of a deep copy shouldn't matter in this case.  And it's easier to just do
// the conversion than get involved in the internals's of SWIG's set wrapper to fix it.

%{
    inline PyObject * convertNameSet(std::set<std::string> const & input) {
        ndarray::PyPtr result(PyTuple_New(input.size()));
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
    // I'll never understand why swig passes pointers to reference typemaps, but it does.
    $result = convertNameSet(*$1);
}

// SWIG doesn't understand Schema::forEach, but the Schema interface provides no other
// way of getting field names in definition order. This forEach functor records field names,
// allowing the python asList Schema method to return schema items in an order consistent
// with forEach.
%{
    struct _FieldNameExtractor {
        std::vector<std::string> mutable * _vec;

        _FieldNameExtractor(std::vector<std::string> * vec) : _vec(vec) { }

        template <typename T>
        void operator()(lsst::afw::table::SchemaItem<T> const & item) const {
            _vec->push_back(item.field.getName());
        }
    };
%}

// ---------------------------------------------------------------------------------------------------------

// SchemaItem objects will often be temporaries, where we only want one of their members, as in
// schema.find("field.name").key.  That causes problems, because swig returns members by pointer
// or by reference, and that reference dangles since the SchemaItem is a temporary.
// In C++ it's safe, because the language guarantees that the temporaries won't be destroyed until the
// end of the statement, but in Python it's guaranteed undefined behavior.

// To get around this, we don't let swig see the innards of the SchemaItem definition (it's #ifdef'd out),
// and we replace it with the following:

%extend lsst::afw::table::SchemaItem {
    // these force return-by-value
    lsst::afw::table::Field< T > getField() const { return self->field; }
    lsst::afw::table::Key< T > getKey() const { return self->key; }
    // now some properties to make the Python interface look like the C++ one
    %pythoncode %{
        field = property(getField)
        key = property(getKey)
    %}
}

// =============== AliasMap =================================================================================

%include "std_map.i"
%include "std_pair.i"
%include "lsst/afw/utils.i"
%template(AliasMapPair) std::pair<std::string,std::string>;
%template(AliasMapInternal) std::map<std::string,std::string>;
%shared_ptr(lsst::afw::table::AliasMap);
%include "lsst/afw/table/AliasMap.h"
%useValueEquality(lsst::afw::table::AliasMap)
%extend lsst::afw::table::AliasMap {
%pythoncode %{

def iteritems(self):
    i = self.begin()
    end = self.end()
    while i != end:
        yield i.value()
        i.incr()
def iterkeys(self):
    for k, v in self.iteritems():
        yield k
def itervalues(self):
    for k, v in self.iteritems():
        yield v
def __iter__(self):
    return self.iterkeys()

def items(self): return list(self.iteritems())
def keys(self): return list(self.iterkeys())
def values(self): return list(self.itervalues())

def __getitem__(self, alias): return self.get(alias)
def __setitem__(self, alias, target): self.set(alias, target)
def __delitem__(self, alias):
    if not self.erase(alias):
        raise KeyError(alias)
def __len__(self): return self.size()
def __nonzero__(self): return not self.empty()
%}
}

// =============== Schemas and their components =============================================================

%include "lsst/afw/table/FieldBase.h"
%include "lsst/afw/table/Field.h"
%include "lsst/afw/table/KeyBase.h"
%include "lsst/afw/table/Key.h"
%include "lsst/afw/table/detail/SchemaImpl.h"


%rename("__eq__") lsst::afw::table::Schema::operator==;
%rename("__ne__") lsst::afw::table::Schema::operator!=;
%rename("__getitem__") lsst::afw::table::Schema::operator[];
%rename("__getitem__") lsst::afw::table::SubSchema::operator[];
%addStreamRepr(lsst::afw::table::Schema)
%include "lsst/afw/table/Schema.h"

%extend lsst::afw::table::SchemaItem {
%pythoncode %{
    def __getitem__(self, i):
        if i == 0:
            return self.key
        elif i == 1:
            return self.field
        raise IndexError("SchemaItem index must be 0 or 1")
    def __str__(self):
        return str(tuple(self))
    def __repr__(self):
        return "SchemaItem(%r, %r)" % (self.key, self.field)
%}
}

%extend lsst::afw::table::Schema {

    void reset(lsst::afw::table::Schema & other) { *self = other; }

    std::vector<std::string> getOrderedNames() {
        std::vector<std::string> names;
        self->forEach(_FieldNameExtractor(&names));
        return names;
    }

%pythoncode %{

extract = _syntax.Schema_extract

def asList(self):
    # This should be replaced by an implementation that uses Schema::forEach directly
    # if/when SWIG gets better at handling templates or we switch to Boost.Python.
    result = []
    for name in self.getOrderedNames():
        result.append(self.find(name))
    return result

def __iter__(self):
    return iter(self.asList())

def __contains__(self, k):
    try:
        r = self.find(k)
        return True
    except:
        return False

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
         except (lsst.pex.exceptions.TypeError, lsst.pex.exceptions.NotFoundError):
             pass
    raise KeyError("Field '%s' not found in Schema." % k)

def addField(self, field, type=None, doc="", units="", size=None, doReplace=False):
    if type is None:
        try:
            prefix, suffix = __builtins__['type'](field).__name__.split("_")
        except Exception:
            raise TypeError("First argument to Schema.addField must be a Field if 'type' is not given.")
        if prefix != "Field":
            raise TypeError("First argument to Schema.addField must be a Field if 'type' is not given.")
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
    return method(field, doc, units, size, doReplace)

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
         except (lsst.pex.exceptions.TypeError, lsst.pex.exceptions.NotFoundError):
             pass
    raise KeyError("Field '%s' not found in Schema." % self.getPrefix())

def asField(self):
    for suffix in _suffixes.itervalues():
         attr = "_asField_" + suffix
         method = getattr(self, attr)
         try:
             return method()
         except (lsst.pex.exceptions.TypeError, lsst.pex.exceptions.NotFoundError):
             pass
    raise KeyError("Field '%s' not found in Schema." % self.getPrefix())

def asKey(self):
    for suffix in _suffixes.itervalues():
         attr = "_asKey_" + suffix
         method = getattr(self, attr)
         try:
             return method()
         except (lsst.pex.exceptions.TypeError, lsst.pex.exceptions.NotFoundError):
             pass
    raise KeyError("Field '%s' not found in Schema." % self.getPrefix())

%}
} // %extend SubSchema

%ignore lsst::afw::table::SchemaMapper::operator=;

%include "lsst/afw/table/SchemaMapper.h"

%template(SchemaVector) std::vector<lsst::afw::table::Schema>;
%template(SchemaMapperVector) std::vector<lsst::afw::table::SchemaMapper>;

// =============== FunctorKeys ==============================================================================

// We need the extend blocks in the macros below because we've removed the FunctorKey overloads of
// get() and set() in the .h file via an #ifdef SWIG.  We had to do that to avoid instantiating the
// FunctorKey overloads for the same types we instantiate the non-FunctorKey overloads (when you use
// Swig's %template statement on a templated function or member function, if instantiates all
// overloads).

%include "lsst/afw/table/FunctorKey.h"

%define %declareOutputFunctorKey(PYNAME, U...)
%shared_ptr(lsst::afw::table::OutputFunctorKey< U >);
%template(OutputFunctorKey ## PYNAME) lsst::afw::table::OutputFunctorKey< U >;
%enddef

%define %declareInputFunctorKey(PYNAME, U...)
%shared_ptr(lsst::afw::table::InputFunctorKey< U >);
%template(InputFunctorKey ## PYNAME) lsst::afw::table::InputFunctorKey< U >;
%enddef

%define %declareFunctorKey(PYNAME, U...)
%declareOutputFunctorKey(PYNAME, U)
%declareInputFunctorKey(PYNAME, U)
%shared_ptr(lsst::afw::table::FunctorKey< U >);
%template(FunctorKey ## PYNAME) lsst::afw::table::FunctorKey< U >;
%enddef

%define %declareReferenceFunctorKey(PYNAME, U...)
%shared_ptr(lsst::afw::table::ReferenceFunctorKey< U >);
%nodefaultctor lsst::afw::table::ReferenceFunctorKey< U >;
%template(ReferenceFunctorKey ## PYNAME) lsst::afw::table::ReferenceFunctorKey< U >;
%enddef

%define %declareConstReferenceFunctorKey(PYNAME, U...)
%shared_ptr(lsst::afw::table::ConstReferenceFunctorKey< U >);
%nodefaultctor lsst::afw::table::ConstReferenceFunctorKey< U >;
%template(ConstReferenceFunctorKey ## PYNAME) lsst::afw::table::ConstReferenceFunctorKey< U >;
%enddef

// =============== BaseTable and BaseRecord =================================================================

%shared_ptr(lsst::afw::table::BaseTable);
%shared_ptr(lsst::afw::table::BaseRecord);

%include "lsst/afw/table/BaseTable.h"

%extend lsst::afw::table::BaseTable {
    %pythoncode %{
        schema = property(getSchema)
        def cast(self, type_):
            return type_._cast(self)
    %}
}

%ignore lsst::afw::table::BaseRecord::operator=;
%include "lsst/afw/table/BaseRecord.h"

%extend lsst::afw::table::BaseRecord {
    %pythoncode %{
        extract = _syntax.BaseRecord_extract
        table = property(lambda self: self.getTable()) # extra lambda allows for polymorphism in property
        schema = property(getSchema)
        def cast(self, type_):
            return type_._cast(self)
    %}
    %feature("shadow") __getitem__ %{
    def __getitem__(self, key):
        if isinstance(key, basestring):
            return self[self.schema.find(key).key]
        try:
            return $action(self, key)
        except NotImplementedError:
            # If this doesn't work as a regular key, try it as a FunctorKey
            return key.get(self)
    %}
    %pythonprepend __setitem__ %{
        if isinstance(args[0], basestring):
            self[self.schema.find(args[0]).key] = args[1]
            return
    %}
    // Allow field name strings be used in place of keys (but only in Python)
    %feature("shadow") get %{
    def get(self, key):
        if isinstance(key, basestring):
            return self.get(self.schema.find(key).key)
        try:
            return $action(self, key)
        except NotImplementedError:
            # If this doesn't work as a regular key, try it as a FunctorKey
            return key.get(self)
    %}
    %feature("shadow") set %{
    def set(self, key, value):
        if isinstance(key, basestring):
            self.set(self.schema.find(key).key, value)
            return
        try:
            $action(self, key, value)
        except NotImplementedError:
            # If this doesn't work as a regular key, try it as a FunctorKey
            return key.set(self, value)
    %}

}

%addCastMethod(lsst::afw::table::BaseTable, lsst::afw::table::BaseTable)
%addCastMethod(lsst::afw::table::BaseRecord, lsst::afw::table::BaseRecord)
%usePointerEquality(lsst::afw::table::BaseRecord)
%usePointerEquality(lsst::afw::table::BaseTable)

// =============== BaseColumnView ===========================================================================

%template(FlagKeyVector) std::vector< lsst::afw::table::Key< lsst::afw::table::Flag > >;

%feature("shadow") lsst::afw::table::BaseColumnView::getBits %{
def getBits(self, keys=None):
    if keys is None:
        return self.getAllBits()
    arg = FlagKeyVector()
    for k in keys:
        if isinstance(k, basestring):
            arg.append(self.schema.find(k).key)
        else:
            arg.append(k)
    return $action(self, arg)
%}

%include "lsst/afw/table/BaseColumnView.h"

%extend lsst::afw::table::BitsColumn {
    PyObject * getSchemaItems() const {
        // Can't use SWIG's std::vector wrapper because SchemaItem doesn't have a default
        // ctor.  And we want to return a list anyway, so you can print it easily and not
        // worry about dangling references and implicit const-casts.
        PyObject * result = PyList_New(0);
        typedef std::vector< lsst::afw::table::SchemaItem<lsst::afw::table::Flag> > ItemVector;
        for (
            ItemVector::const_iterator i = self->getSchemaItems().begin();
            i != self->getSchemaItems().end();
            ++i
        ) {
            PyObject * pyItem = SWIG_NewPointerObj(
                new lsst::afw::table::SchemaItem<lsst::afw::table::Flag>(*i),
                SWIGTYPE_p_lsst__afw__table__SchemaItemT_lsst__afw__table__Flag_t,
                true // SWIG takes ownership of the pointer
            );
            if (!pyItem) {
                Py_DECREF(result);
                return NULL;
            }
            if (PyList_Append(result, pyItem) != 0) {
                Py_DECREF(result);
                Py_DECREF(pyItem);
                return NULL;
            }
        }
        return result;
    }
    %pythoncode %{
        array = property(getArray)
    %}
}

%extend lsst::afw::table::BaseColumnView {
    %pythoncode %{
        extract = _syntax.BaseColumnView_extract
        table = property(getTable)
        schema = property(getSchema)
        def get(self, key):
            """Return the column for the given key or field name; synonym for __getitem__."""
            return self[key]
    %}
    // Allow field name strings be used in place of keys (but only in Python)
    %pythonprepend __getitem__ %{
        if isinstance(args[0], basestring):
            return self[self.schema.find(args[0]).key]
    %}
}

// =============== Field Types ==============================================================================

// Must come after the FlagKeyVector %template, or SWIG bungles the generated code.
%include "lsst/afw/table/Flag.h"

%pythoncode %{
from ..geom import Angle, Point2D, Point2I
from ..geom.ellipses import Quadrupole
from ..coord import Coord, IcrsCoord
import numpy
Field = {}
Key = {}
SchemaItem = {}
_suffixes = {}
aliases = {
    int: "I",
    long: "L",
    float: "D",
    str: "String",
    numpy.uint16: "U",
    numpy.int32: "I",
    numpy.int64: "L",
    numpy.float32: "F",
    numpy.float64: "D",
    Angle: "Angle",
    Coord: "Coord",
    IcrsCoord: "Coord",
    Point2I: "PointI",
    Point2D: "PointD",
    Quadrupole: "MomentsD",
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

%declareFieldType(boost::uint16_t, U)
%declareFieldType(boost::int32_t, I)
%declareFieldType(boost::int64_t, L)
%declareFieldType(float, F)
%declareFieldType(double, D)
%declareFieldType(std::string, String)
%declareFieldType(lsst::afw::table::Flag, Flag)
%declareFieldType(lsst::afw::geom::Angle, Angle)
%declareFieldType(lsst::afw::coord::Coord, Coord)

%declareFieldType(lsst::afw::table::Point<boost::int32_t>, PointI)
%declareFieldType(lsst::afw::table::Point<double>, PointD)

%declareFieldType(lsst::afw::table::Moments<double>, MomentsD)

%declareFieldType(lsst::afw::table::Array<boost::uint16_t>, ArrayU)
%declareFieldType(lsst::afw::table::Array<int>, ArrayI)
%declareFieldType(lsst::afw::table::Array<float>, ArrayF)
%declareFieldType(lsst::afw::table::Array<double>, ArrayD)

%declareFieldType(lsst::afw::table::Covariance<float>, CovF)

%declareFieldType(lsst::afw::table::Covariance< lsst::afw::table::Point<float> >, CovPointF)

%declareFieldType(lsst::afw::table::Covariance< lsst::afw::table::Moments<float> >, CovMomentsF)

%include "lsst/afw/table/specializations.i"

%pythoncode %{
# underscores here prevent these from becoming global names
for _d in (Field, Key, SchemaItem, _suffixes):
    for _k, _v in aliases.iteritems():
        _d[_k] = _d[_v]
%}

// =============== Catalogs =================================================================================

%include "lsst/afw/table/Catalog.i"

namespace lsst { namespace afw { namespace table {

%declareCatalog(CatalogT, Base)

}}} // namespace lsst::afw::table
