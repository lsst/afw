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
 
%define imageLib_DOCSTRING
"
Basic routines to talk to lsst::afw::image classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.image", docstring=imageLib_DOCSTRING) imageLib

// Suppress swig complaints
#pragma SWIG nowarn=314                 // print is a python keyword (--> _print)
#pragma SWIG nowarn=362                 // operator=  ignored


%{
#include "boost/cstdint.hpp"

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging.h"
#include "lsst/pex/policy.h"
#include "lsst/afw/cameraGeom.h"
#include "lsst/afw/image.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/image/Color.h"
#include "lsst/afw/fits.h" // just for exceptions

#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_IMAGE_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "ndarray/swig.h"
#include "ndarray/swig/eigen.h"

#include "lsst/afw/formatters/WcsFormatter.h"
#include "lsst/afw/formatters/TanWcsFormatter.h"
#include "lsst/afw/formatters/ExposureFormatter.h"
#include "lsst/afw/formatters/DecoratedImageFormatter.h"

#pragma clang diagnostic ignored "-Warray-bounds" // PyTupleObject has an array declared as [1]
%}

%include "../boost_picklable.i"

%init %{
    import_array();
%}

/************************************************************************************************************/

%typemap(typecheck, precedence=SWIG_TYPECHECK_BOOL, noblock=1) bool {
    $1 = PyBool_Check($input) ? 1 : 0;
}

%include "lsst/p_lsstSwig.i"
%include "lsst/daf/base/persistenceMacros.i"

%include "ndarray.i"

%declareNumPyConverters(Eigen::MatrixXd);
%declareNumPyConverters(Eigen::VectorXd);
%declareNumPyConverters(Eigen::Matrix2d);
%declareNumPyConverters(Eigen::Vector2d);
%declareNumPyConverters(Eigen::Matrix3d);
%declareNumPyConverters(Eigen::Vector3d);

%declareNumPyConverters(ndarray::Array<unsigned short,2,1>);
%declareNumPyConverters(ndarray::Array<unsigned short const,2,1>);

%declareNumPyConverters(ndarray::Array<int,2,1>);
%declareNumPyConverters(ndarray::Array<int const,2,1>);

%declareNumPyConverters(ndarray::Array<float,2,1>);
%declareNumPyConverters(ndarray::Array<float const,2,1>);

%declareNumPyConverters(ndarray::Array<double,2,1>);
%declareNumPyConverters(ndarray::Array<double const,2,1>);

%lsst_exceptions();

/************************************************************************************************************/

%pythoncode {
    def _getBBoxFromSliceTuple(img, imageSlice):
        """Given a slice specification return the proper Box2I
    This is the worker routine behind __getitem__ and __setitem__

    The imageSlice may be:
       lsst.afw.geom.Box2I
       slice, slice
       :
    Only the first one or two parts of the slice are recognised (no stride), a single int is
    interpreted as n:n+1, and negative indices are interpreted relative to the end of the array,
    so supported slices include:
       2
       -1
       1:10
       :-2
       : (equivalent to ... (python's Ellipsis) which is also supported)

    E.g.
     im[-1, :]
     im[..., 18]
     im[4,  10]
     im[-3:, -2:]
     im[-2, -2]
     im[1:4, 6:10]
     im[:]
    """
        afwGeom = lsst.afw.geom.geomLib

        if isinstance(imageSlice, afwGeom.Box2I):
            return imageSlice

        if isinstance(imageSlice, slice) and imageSlice.start is None and imageSlice.stop is None:
            imageSlice = (Ellipsis, Ellipsis,)

        if not (isinstance(imageSlice, tuple) and len(imageSlice) == 2 and \
                    sum([isinstance(_, (slice, type(Ellipsis), int)) for _ in imageSlice]) == 2):
            raise IndexError("Images may only be indexed as a 2-D slice not %s",
                             imageSlice[0], imageSlice[1])
        
        imageSlice, _imageSlice = [], imageSlice
        for s, wh in zip(_imageSlice, img.getDimensions()):
            if isinstance(s, slice):
                pass
            elif isinstance(s, int):
                if s < 0:
                    s += wh
                s = slice(s, s + 1)
            else:
                s = slice(0, wh)

            imageSlice.append(s)

        x, y = [_.indices(wh) for _, wh in zip(imageSlice, img.getDimensions())]
        return afwGeom.Box2I(afwGeom.Point2I(x[0], y[0]), afwGeom.Point2I(x[1] - 1, y[1] - 1))
}

%{
#include "lsst/afw/math/Function.h"
#include "lsst/afw/math/FunctionLibrary.h"
%}
%import "lsst/afw/math/function.i"

%include "image.i"
%include "ImagePca.i"
%include "Utils.i"
%include "mask.i"
%include "maskedImage.i"
%include "imageSlice.i"

%apply double &OUTPUT { double & };
%rename(positionToIndexAndResidual) lsst::afw::image::positionToIndex(double &, double);
%clear double &OUTPUT;

%include "lsst/afw/image/ImageUtils.h"

%include "lsst/afw/image/Wcs.i"
%include "lsst/afw/image/TanWcs.i"

/************************************************************************************************************/

#if !defined(CAMERA_GEOM_LIB_I)
%import "lsst/afw/cameraGeom/cameraGeomLib.i"
#endif

%include "lsst/afw/image/Color.h"
%include "lsst/afw/image/Defect.i"
%include "lsst/afw/image/Exposure.i"
