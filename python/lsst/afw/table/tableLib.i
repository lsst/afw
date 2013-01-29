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

%include "lsst/afw/table/table_fwd.i"

%lsst_exceptions();

//---------- Dependencies that don't need to be seen by downstream imports ----------------------------------

%import "lsst/afw/fits/fitsLib.i"
%import "lsst/daf/base/baseLib.i"
%import "lsst/afw/coord/coord_fwd.i"
%import "lsst/afw/geom/ellipses/ellipses_fwd.i"
%import "lsst/afw/geom/geom_fwd.i"

// Wcs, Calib, Psf, and Footprint are needed by SourceRecord, ExposureRecord, but we don't want to %import
// them, for fear of circular dependencies.
// Happily, forward declarations and %shared_ptr are all we need.
namespace lsst { namespace afw {
namespace image {
    class Wcs;
    class Calib;
}
namespace detection {
    class Psf;
    class Footprint;
}}}
%shared_ptr(lsst::afw::image::Wcs);
%shared_ptr(lsst::afw::image::Calib);
%shared_ptr(lsst::afw::detection::Psf);
%shared_ptr(lsst::afw::detection::Footprint);

//---------- ndarray and Eigen NumPy conversion typemaps ----------------------------------------------------

%{
#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_TABLE_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "ndarray/swig.h"
#include "ndarray/swig/eigen.h"
#include "lsst/afw/geom/Angle.h"

// This enables numpy array conversion for Angle, converting it to a regular array of double.
namespace ndarray { namespace detail {
template <> struct NumpyTraits<lsst::afw::geom::Angle> : public NumpyTraits<double> {};
}}

%}
%include "ndarray.i"

%init %{
    import_array();
%}

%declareNumPyConverters(ndarray::Array<bool const,1>);
%declareNumPyConverters(ndarray::Array<lsst::afw::table::RecordId const,1>);
%declareNumPyConverters(ndarray::Array<boost::int32_t const,1>);
%declareNumPyConverters(ndarray::Array<boost::int64_t const,1>);
%declareNumPyConverters(ndarray::Array<boost::int32_t,1>);
%declareNumPyConverters(ndarray::Array<boost::int64_t,1>);
%declareNumPyConverters(ndarray::Array<boost::int32_t,1,1>);
%declareNumPyConverters(ndarray::Array<boost::int64_t,1,1>);
%declareNumPyConverters(ndarray::Array<boost::int32_t const,1,1>);
%declareNumPyConverters(ndarray::Array<boost::int64_t const,1,1>);
%declareNumPyConverters(ndarray::Array<float,1>);
%declareNumPyConverters(ndarray::Array<double,1>);
%declareNumPyConverters(ndarray::Array<float const,1>);
%declareNumPyConverters(ndarray::Array<double const,1>);
%declareNumPyConverters(ndarray::Array<float,1,1>);
%declareNumPyConverters(ndarray::Array<double,1,1>);
%declareNumPyConverters(ndarray::Array<float const,1,1>);
%declareNumPyConverters(ndarray::Array<double const,1,1>);
%declareNumPyConverters(ndarray::Array<float,2>);
%declareNumPyConverters(ndarray::Array<double,2>);
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

//---------- STL Typemaps and Template Instantiations -------------------------------------------------------

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

//---------- afw::table classes and functions ---------------------------------------------------------------

%pythoncode %{
from . import _syntax
%}

%include "Base.i"
%include "Simple.i"
%include "Source.i"
%include "Match.i"
%include "Exposure.i"
