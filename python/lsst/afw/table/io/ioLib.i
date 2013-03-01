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
 
%define ioLib_DOCSTRING
"
Python interface to lsst::afw::table::io classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.table.io", docstring=ioLib_DOCSTRING) ioLib

#pragma SWIG nowarn=362                 // operator=  ignored
#pragma SWIG nowarn=389                 // operator[]  ignored
#pragma SWIG nowarn=503                 // comparison operators ignored
#pragma SWIG nowarn=520                 // base class not similarly marked as smart pointer
#pragma SWIG nowarn=401                 // nothing known about base class
#pragma SWIG nowarn=302                 // redefine identifier (SourceSet<> -> SourceSet)

%{
#include "lsst/afw/table/io/Persistable.h"
%}

%include "boost_shared_ptr.i"
%include "lsst/p_lsstSwig.i"
%import "lsst/pex/exceptions/exceptionsLib.i"

%lsst_exceptions();

// =============== Persistable ==============================================================================

%shared_ptr(lsst::afw::table::io::Persistable);

%define %declareTablePersistable(NAME, T)
%shared_ptr(lsst::afw::table::io::Persistable);
%shared_ptr(lsst::afw::table::io::PersistableFacade< T >);
%shared_ptr(T);
%template(NAME ## PersistableFacade) lsst::afw::table::io::PersistableFacade< T >;
%enddef

%include "lsst/afw/table/io/Persistable.h"

// =============== ModuleImporter ===========================================================================

%{
#include "lsst/afw/table/io/ModuleImporter.h"

namespace lsst { namespace afw { namespace table { namespace io {
namespace {
class PythonModuleImporter : public ModuleImporter {
public:
    static ModuleImporter const * get() {
        static PythonModuleImporter const instance;
        return &instance;
    }
private:
    PythonModuleImporter() {}
protected:
    virtual bool _import(std::string const & name) const;
};

bool PythonModuleImporter::_import(std::string const & name) const {
    PyObject * mod = PyImport_ImportModule(name.c_str());
    if (mod) {
        Py_DECREF(mod);
        return true;
    }
    return false;
}

} // anonymous
}}}} // namespace lsst::afw::table::io
%}

%init %{
    lsst::afw::table::io::ModuleImporter::install(lsst::afw::table::io::PythonModuleImporter::get());
%}
