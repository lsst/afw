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
%{
#include "lsst/afw/image/Calib.h"
%}

%import "lsst/afw/table/io/ioLib.i"

%declareTablePersistable(Calib, lsst::afw::image::Calib);

%typemap(out) std::pair<ndarray::Array<double,1>,ndarray::Array<double,1> > {
    $result = PyTuple_New(2);
    {
        // need to invoke SwigValueWrapper's conversion operator
        std::pair<ndarray::Array<double,1>,ndarray::Array<double,1> > & pair = $1;
        PyObject * a0 = ndarray::PyConverter< ndarray::Array<double,1> >::toPython(pair.first);
        PyObject * a1 = ndarray::PyConverter< ndarray::Array<double,1> >::toPython(pair.second);
        PyTuple_SET_ITEM($result, 0, a0);
        PyTuple_SET_ITEM($result, 1, a1);
    }
}

%include "lsst/afw/image/Calib.h"
%template(vectorCalib) std::vector<boost::shared_ptr<const lsst::afw::image::Calib> >;

%extend lsst::afw::image::Calib {
    %pythoncode %{
        #
        # Provide return value for C++ "void operator op=()" or it will magically end up as None
        #
        def __imul__(*args):
            """__imul__(self, double scale) -> self"""
            _imageLib.Calib___imul__(*args)
            return args[0]
    
        def __idiv__(*args):
            """__idiv__(self, double scale) -> self"""
            _imageLib.Calib___idiv__(*args)
            return args[0]
    %}
}
