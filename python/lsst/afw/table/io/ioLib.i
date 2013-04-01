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

// =============== Utility code =============================================================================

%inline %{

// It's useful in test code to be able to compare Persistables for pointer equality,
// but I don't think this is possible without actually wrapping a shared_ptr equality
// comparison - the Swig 'this' objects only expose the address *of* the shared_ptr,
// not the address *in* the shared_ptr.
bool comparePersistablePtrs(
    PTR(lsst::afw::table::io::Persistable) a, PTR(lsst::afw::table::io::Persistable) b
) {
    return a == b;
}

%}
