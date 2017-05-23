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

#include <ostream>
#include <sstream>
#include <string>
#include <memory>
#include <vector>

#include "astshim.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/SpherePoint.h"
#include "lsst/afw/geom/Endpoint.h"

namespace lsst {
namespace afw {
namespace geom {
namespace {

/*
Get a pointer to a frame

If frame is a FrameSet then return a copy of its current frame, else return the original argument
*/
std::shared_ptr<ast::Frame> getCurrentFrame(std::shared_ptr<ast::Frame> framePtr) {
    auto frameSetPtr = std::dynamic_pointer_cast<ast::FrameSet>(framePtr);
    if (frameSetPtr) {
        return frameSetPtr->getFrame(ast::FrameSet::CURRENT);
    }
    return framePtr;
}

}  // <anonymous>

template <typename Point, typename Array>
BaseEndpoint<Point, Array>::BaseEndpoint(int nAxes) : _nAxes(nAxes) {
    if (nAxes <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "nAxes = " + std::to_string(nAxes) + "; must be > 0");
    }
}

template <typename Point, typename Array>
std::shared_ptr<ast::Frame> BaseEndpoint<Point, Array>::makeFrame() const {
    return std::make_shared<ast::Frame>(getNAxes());
}

template <typename Point, typename Array>
void BaseEndpoint<Point, Array>::_assertNAxes(int nAxes) const {
    if (nAxes != this->getNAxes()) {
        std::ostringstream os;
        os << "number of axes provided " << nAxes << " != " << this->getNAxes() << " required";
        throw std::invalid_argument(os.str());
    }
}

template <typename Point>
std::vector<double> BaseVectorEndpoint<Point>::dataFromPoint(Point const& point) const {
    const int nAxes = this->getNAxes();
    std::vector<double> result(nAxes);
    for (int axInd = 0; axInd < nAxes; ++axInd) {
        result[axInd] = point[axInd];
    }
    return result;
}

template <typename Point>
ndarray::Array<double, 2, 2> BaseVectorEndpoint<Point>::dataFromArray(Array const& arr) const {
    const int nAxes = this->getNAxes();
    const int nPoints = this->getNPoints(arr);
    ndarray::Array<double, 2, 2> data = ndarray::allocate(ndarray::makeVector(nAxes, nPoints));
    auto dataColIter = data.transpose().begin();
    for (auto const& point : arr) {
        for (int axInd = 0; axInd < nAxes; ++axInd) {
            (*dataColIter)[axInd] = point[axInd];
        }
        ++dataColIter;
    }
    return data;
}

template <typename Point>
Point BaseVectorEndpoint<Point>::pointFromData(std::vector<double> const& data) const {
    this->_assertNAxes(this->_getNAxes(data));
    return Point(data.data());
}

std::vector<double> GenericEndpoint::dataFromPoint(Point const& point) const {
    this->_assertNAxes(_getNAxes(point));
    return point;
}

ndarray::Array<double, 2, 2> GenericEndpoint::dataFromArray(Array const& arr) const {
    this->_assertNAxes(_getNAxes(arr));
    return ndarray::copy(arr);
}

std::vector<double> GenericEndpoint::pointFromData(std::vector<double> const& data) const {
    this->_assertNAxes(data.size());
    return data;
}

ndarray::Array<double, 2, 2> GenericEndpoint::arrayFromData(ndarray::Array<double, 2, 2> const& data) const {
    this->_assertNAxes(_getNAxes(data));
    return ndarray::copy(data);
}

Point2Endpoint::Point2Endpoint(int nAxes) : BaseVectorEndpoint<Point2D>(2) {
    if (nAxes != 2) {
        std::ostringstream os;
        os << "nAxes = " << nAxes << " != 2";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
}

std::vector<Point2D> Point2Endpoint::arrayFromData(ndarray::Array<double, 2, 2> const& data) const {
    this->_assertNAxes(this->_getNAxes(data));
    int const nPoints = this->_getNPoints(data);
    Array array;
    array.reserve(nPoints);
    for (auto const& dataCol : data.transpose()) {
        array.emplace_back(dataCol[0], dataCol[1]);
    }
    return array;
}

void Point2Endpoint::normalizeFrame(std::shared_ptr<ast::Frame> framePtr) const {
    // use getCurrentFrame because if framePtr points to a FrameSet we want the name of its current frame
    std::string className = getCurrentFrame(framePtr)->getClassName();
    if (className != "Frame") {
        std::ostringstream os;
        os << "frame is a " << className << ", not a Frame";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
}

SpherePointEndpoint::SpherePointEndpoint(int nAxes) : BaseVectorEndpoint(2) {
    if (nAxes != 2) {
        std::ostringstream os;
        os << "nAxes = " << nAxes << " != 2";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
}

std::vector<SpherePoint> SpherePointEndpoint::arrayFromData(ndarray::Array<double, 2, 2> const& data) const {
    this->_assertNAxes(this->_getNAxes(data));
    int const nPoints = this->_getNPoints(data);
    Array array;
    array.reserve(nPoints);
    for (auto const& dataCol : data.transpose()) {
        array.emplace_back(dataCol[0] * radians, dataCol[1] * radians);
    }
    return array;
}

std::shared_ptr<ast::Frame> SpherePointEndpoint::makeFrame() const {
    return std::make_shared<ast::SkyFrame>();
}

void SpherePointEndpoint::normalizeFrame(std::shared_ptr<ast::Frame> framePtr) const {
    // use getCurrentFrame because if framePtr points to a FrameSet we want its current frame
    auto currentFramePtr = getCurrentFrame(framePtr);
    auto skyFramePtr = std::dynamic_pointer_cast<ast::SkyFrame>(currentFramePtr);
    if (!skyFramePtr) {
        std::ostringstream os;
        os << "frame is a " << currentFramePtr->getClassName() << ", not a SkyFrame";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    if (skyFramePtr->getLonAxis() != 1) {
        // axes are swapped to Lat, Lon; swap them back to the usual Lon, Lat
        // warning: be sure to call permAxes on the original framePtr argument,
        // as otherwise it will have no effect if framePtr points to a FrameSet
        std::vector<int> perm = {2, 1};
        framePtr->permAxes(perm);
    }
}

std::ostream& operator<<(std::ostream& os, GenericEndpoint const& endpoint) {
    os << "GenericEndpoint(" << endpoint.getNAxes() << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, Point2Endpoint const& endpoint) {
    os << "Point2Endpoint()";
    return os;
}

std::ostream& operator<<(std::ostream& os, SpherePointEndpoint const& endpoint) {
    os << "SpherePointEndpoint()";
    return os;
}

// explicit instantiations
template class BaseEndpoint<std::vector<double>, ndarray::Array<double, 2, 2>>;
template class BaseEndpoint<Point2D, std::vector<Point2D>>;
template class BaseEndpoint<SpherePoint, std::vector<SpherePoint>>;

template class BaseVectorEndpoint<Point2D>;
template class BaseVectorEndpoint<SpherePoint>;

}  // geom
}  // afw
}  // lsst
