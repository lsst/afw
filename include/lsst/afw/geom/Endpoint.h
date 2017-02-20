// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2017 LSST Corporation.
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

#ifndef LSST_AFW_GEOM_ENDPOINT_H
#define LSST_AFW_GEOM_ENDPOINT_H

#include <memory>
#include <vector>

#include "astshim.h"
#include "ndarray.h"

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/SpherePoint.h"

namespace lsst {
namespace afw {
namespace geom {


/**
Virtual base class for endpoints, which are helper classes for Transform

Endpoints transform points and lists of points from LSST-specific data types,
such as Point2D and SpherePoint, to a form accepted by ast::Mapping.tran.
Each type of endpoint is used for a particular LSST data type, for example:
- Point2Endpoint is used for Point2D data
- Point3Endpoint is used for Point3D data
- SpherePointEndpoint for SpherePoint data
- GenericEndpoint is used when no other form will do; its LSST data type
  is identical to the type used for ast::Mapping.tran.

Endpoints use the following form of data for ast::Mapping.tran
(other forms are possible, but this is the one chosen):
- std::vector<double> for a single point
- ndarray<double, 2, 2> for an array of points

Endpoints are designed as helper classes for Transform. Each transform has a two endpoints:
one for input data and one for output data.

Endpoint also provides two methods to work with ast::Frames:
- normalizeFrame verifies that a frame is the correct type, and adjusts its settings if necessary
- makeFrame creates a new frame with the correct type and settings

@tparam PointT  LSST data type for one point
@tparam ArrayT  LSST data type for an array of points
*/

template <typename PointT, typename ArrayT>
class BaseEndpoint {
public:
    using Point = PointT;
    using Array = ArrayT;

    BaseEndpoint(BaseEndpoint const &) = default;
    BaseEndpoint(BaseEndpoint &&) = default;
    BaseEndpoint & operator=(BaseEndpoint const &) = default;
    BaseEndpoint & operator=(BaseEndpoint &&) = default;

    virtual ~BaseEndpoint() {};

    int getNAxes() const { return _nAxes; }

    /**
    Return the number of points in an array
    */
    virtual int getNPoints(Array const &arr) const = 0;

    /**
    Get raw data from a single point

    @param[in] point  data for a single point
    @return the values in the point as a vector of size NAxess

    @throw lsst::pex::exceptions::InvalidParameterError if the point has the wrong number of axes
    */
    virtual std::vector<double> dataFromPoint(Point const & point) const = 0;

    /**
    Get raw data from an array of points
    
    @param[in] arr  Array of points
    @return the data as a 2-D ndarray array [nPoints, nAxes] in C order,
        so the in-memory view is, for example, x0, y0, x1, y1, x2, y2, ...

    @throw lsst::pex::exceptions::InvalidParameterError if the array has the wrong nAxes dimension
    */
    virtual ndarray::Array<double, 2, 2> dataFromArray(Array const & arr) const = 0;

    /**
    Get a single point from raw data

    @param[in] data  Data as a vector of length NAxes
    @return the corresponding point
    */
    virtual Point pointFromData(std::vector<double> const & data) const = 0;

    /**
    Get an array of points from raw data
    
    @param[in] data  Raw data for an array of points, as a 2-D ndarray array [nPoints, nAxes] in C order,
        so the in-memory view is, for example, x0, y0, x1, y1, x2, y2, ...
    @return an array of points

    @throw lsst::pex::exceptions::InvalidParameterError if the array has the wrong nAxes dimension
    */
    virtual Array arrayFromData(ndarray::Array<double, 2, 2> const & data) const = 0;

    /**
    Create a Frame that can be used with this end point in a Transform
    */
    virtual std::shared_ptr<ast::Frame> makeFrame() const;

    /**
    Adjust and check the frame as needed.

    Do not obother to check the number of axes because that is done elsewhere.

    The base implementation does nothing.
    */
    virtual void normalizeFrame(std::shared_ptr<ast::Frame> framePtr) const {};

protected:
    /**
    Construct a BaseEndpoint

    @param[in] nAxes  the number of axes in a point
    */
    explicit BaseEndpoint(int nAxes) : _nAxes(nAxes) {}

    void _assertNAxes(int nAxes) const;

    int _getNAxes(ndarray::Array<double, 2, 2> const & data) const {
        return data.getSize<1>();
    }

    int _getNAxes(ndarray::Array<double, 1, 1> const & data) const {
        return data.getSize<0>();
    }

    int _getNAxes(std::vector<double> const & data) const {
        return data.size();
    }

    int _getNPoints(ndarray::Array<double, 2, 2> const & data) const {
        return data.getSize<0>();
    }

private:
    int _nAxes;  /// number of axes
};

/**
Base class for endpoints with Array = std::vector<Point>
*/
template <typename PointT>
class BaseVectorEndpoint : public BaseEndpoint<PointT, std::vector<PointT>> {
public:
    using Array = std::vector<PointT>;
    using Point = PointT;

    BaseVectorEndpoint(BaseVectorEndpoint const &) = default;
    BaseVectorEndpoint(BaseVectorEndpoint &&) = default;
    BaseVectorEndpoint & operator=(BaseVectorEndpoint const &) = default;
    BaseVectorEndpoint & operator=(BaseVectorEndpoint &&) = default;

    virtual ~BaseVectorEndpoint() {};

    virtual int getNPoints(Array const & arr) const override {
        return arr.size();
    }

    virtual std::vector<double> dataFromPoint(Point const & point) const override;

    virtual ndarray::Array<double, 2, 2> dataFromArray(Array const & arr) const override;

    virtual Point pointFromData(std::vector<double> const & data) const override;

    virtual Array arrayFromData(ndarray::Array<double, 2, 2> const & data) const override;

protected:
    /**
    Construct a BaseVectorEndpoint

    @param[in] nAxes  the number of axes in a point
    */
    explicit BaseVectorEndpoint(int nAxes) : BaseEndpoint<Point, Array>(nAxes) {};
};

/**
A generic endpoint for data in the format used by ast::Mapping

Thus supports all ast frame classes and any number of axes, and thus can be used as an endpoint
for any ast::Mapping.
*/
class GenericEndpoint: public BaseEndpoint<std::vector<double>, ndarray::Array<double, 2, 2>> {
public:

    GenericEndpoint(GenericEndpoint const &) = default;
    GenericEndpoint(GenericEndpoint &&) = default;
    GenericEndpoint & operator=(GenericEndpoint const &) = default;
    GenericEndpoint & operator=(GenericEndpoint &&) = default;

    /// Construct a GenericEndpoint with the specified number of axes
    explicit GenericEndpoint(int nAxes) : BaseEndpoint(nAxes) {};

    virtual ~GenericEndpoint() {};

    virtual int getNPoints(Array const & arr) const override {
        return arr.getSize<0>();
    }

    virtual std::vector<double> dataFromPoint(Point const & point) const override;

    virtual ndarray::Array<double, 2, 2> dataFromArray(Array const & arr) const override;

    virtual Point pointFromData(std::vector<double> const & data) const override;

    virtual Array arrayFromData(ndarray::Array<double, 2, 2> const & data) const override;
};


/**
An endpoint for Point<double, N>

@tparam N  number of axes; the only allowed values are 2 or 3
*/
template <int N>
class PointEndpoint : public BaseVectorEndpoint<Point<double, N>> {
public:
    PointEndpoint(PointEndpoint const &) = default;
    PointEndpoint(PointEndpoint &&) = default;
    PointEndpoint & operator=(PointEndpoint const &) = default;
    PointEndpoint & operator=(PointEndpoint &&) = default;

    /// Construct a PointEndpoint; template parameter N specifies the number of axes
    explicit PointEndpoint() : BaseVectorEndpoint<Point<double, N>>(N) {}

    /// This constructor is used by Transform; nAxes must equal template parameter N
    explicit PointEndpoint(int nAxes);

    virtual ~PointEndpoint() {};

    /**
    Check that framePtr points to a Frame, not a subclass

    Subclasses are forbidden because Point<double, N> is assumed to be cartesian
    and subclasses of Frame are not (e.g. SkyFrame, SpecFrame and TimeFrame).
    Note that SpecFrame and TimeFrame are 1-dimensional so they cannot be used
    in any case. A CmpFrame could be cartesian, but we play it safe and reject these
    (however, a cartesian CmpFrame ought to simplify to a Frame).
    */
    virtual void normalizeFrame(std::shared_ptr<ast::Frame> framePtr) const override;
};

// Aliases for the only allowed <N> of PointEndpoint
using Point2Endpoint = PointEndpoint<2>;
using Point3Endpoint = PointEndpoint<3>;

/**
An endpoint for SpherePoint

@warning When SpherePoint is used for celestial WCS the result is likely to obey the FITS convention
that 1,1 is the lower left pixel of the image. If so, you can use an appropriate ast::ShiftMap
that adjusts for both the 1 pixel offset between FITS and LSST convention and the XY0 of the image.
*/
class SpherePointEndpoint : public BaseVectorEndpoint<SpherePoint> {
public:
    SpherePointEndpoint(SpherePointEndpoint const &) = default;
    SpherePointEndpoint(SpherePointEndpoint &&) = default;
    SpherePointEndpoint & operator=(SpherePointEndpoint const &) = default;
    SpherePointEndpoint & operator=(SpherePointEndpoint &&) = default;

    /// Construct a SpherePointEndpoint
    explicit SpherePointEndpoint() : BaseVectorEndpoint(2) {}

    /// This constructor is used by Transform; nAxes must be 2
    explicit SpherePointEndpoint(int nAxes);

    virtual ~SpherePointEndpoint() {};

    /**
    Create a Frame that can be used with this end point in a Transform
    */
    virtual std::shared_ptr<ast::Frame> makeFrame() const override;

    /**
    Check that framePtr points to a SkyFrame and set longitude axis to 0, latitude to 1
    */
    virtual void normalizeFrame(std::shared_ptr<ast::Frame> framePtr) const override;
};

/**
Print "GenericEndpoint(_n_)" to the ostream where `_n_` is the number of axes, e.g. "GenericAxes(4)"
*/
std::ostream & operator<<(std::ostream & os, GenericEndpoint const & endpoint);

/// Print "Point2Endpoint()" to the ostream
std::ostream & operator<<(std::ostream & os, Point2Endpoint const & endpoint);

/// Print "Point3Endpoint()" to the ostream
std::ostream & operator<<(std::ostream & os, Point3Endpoint const & endpoint);

/// Print "SpherePointEndpoint()" to the ostream
std::ostream & operator<<(std::ostream & os, SpherePointEndpoint const & endpoint);

}  // geom
}  // afw
}  // lsst

#endif
