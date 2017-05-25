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

#include <exception>
#include <memory>
#include <ostream>
#include <sstream>
#include <vector>

#include "astshim.h"
#include "lsst/afw/geom/Endpoint.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/SpherePoint.h"
#include "lsst/afw/geom/Transform.h"
#include "lsst/pex/exceptions/Exception.h"

namespace lsst {
namespace afw {
namespace geom {

template <class FromEndpoint, class ToEndpoint>
Transform<FromEndpoint, ToEndpoint>::Transform(ast::Mapping const &mapping, bool simplify)
        : _fromEndpoint(mapping.getNIn()), _frameSet(), _toEndpoint(mapping.getNOut()) {
    auto fromFrame = _fromEndpoint.makeFrame();
    auto toFrame = _toEndpoint.makeFrame();
    if (simplify) {
        _frameSet = std::make_shared<ast::FrameSet>(*fromFrame, *(mapping.simplify()), *toFrame);
    } else {
        _frameSet = std::make_shared<ast::FrameSet>(*fromFrame, mapping, *toFrame);
    }
}

template <class FromEndpoint, class ToEndpoint>
Transform<FromEndpoint, ToEndpoint>::Transform(ast::FrameSet const &frameSet, bool simplify)
        : Transform(simplify ? std::dynamic_pointer_cast<ast::FrameSet>(frameSet.simplify())
                             : frameSet.copy()) {}

template <typename FromEndpoint, typename ToEndpoint>
Transform<FromEndpoint, ToEndpoint>::Transform(std::shared_ptr<ast::FrameSet> &&frameSet)
        : _fromEndpoint(frameSet->getNIn()), _frameSet(frameSet), _toEndpoint(frameSet->getNOut()) {
    // Normalize the base and current frame in a way that affects its behavior as a mapping.
    // To do this one must set the current frame to the frame to be normalized
    // and normalize the frame set as a frame (i.e. normalize the frame "in situ").
    // The obvious alternative of normalizing a shallow copy of the frame does not work;
    // the frame is altered but not the associated mapping!

    // Normalize the current frame by normalizing the frameset as a frame
    _toEndpoint.normalizeFrame(frameSet);

    // Normalize the base frame by temporarily making it the current frame,
    // normalizing the frameset as a frame, then making it the base frame again
    const int currIndex = frameSet->getCurrent();
    const int baseIndex = frameSet->getBase();
    frameSet->setCurrent(baseIndex);
    _fromEndpoint.normalizeFrame(frameSet);
    frameSet->setBase(baseIndex);
    frameSet->setCurrent(currIndex);
}

template <class FromEndpoint, class ToEndpoint>
typename ToEndpoint::Point Transform<FromEndpoint, ToEndpoint>::tranForward(
        typename FromEndpoint::Point const &point) const {
    auto const rawFromData = _fromEndpoint.dataFromPoint(point);
    auto rawToData = _frameSet->tranForward(rawFromData);
    return _toEndpoint.pointFromData(rawToData);
}

template <class FromEndpoint, class ToEndpoint>
typename ToEndpoint::Array Transform<FromEndpoint, ToEndpoint>::tranForward(
        typename FromEndpoint::Array const &array) const {
    auto const rawFromData = _fromEndpoint.dataFromArray(array);
    auto rawToData = _frameSet->tranForward(rawFromData);
    return _toEndpoint.arrayFromData(rawToData);
}

template <class FromEndpoint, class ToEndpoint>
typename FromEndpoint::Point Transform<FromEndpoint, ToEndpoint>::tranInverse(
        typename ToEndpoint::Point const &point) const {
    auto const rawFromData = _toEndpoint.dataFromPoint(point);
    auto rawToData = _frameSet->tranInverse(rawFromData);
    return _fromEndpoint.pointFromData(rawToData);
}

template <class FromEndpoint, class ToEndpoint>
typename FromEndpoint::Array Transform<FromEndpoint, ToEndpoint>::tranInverse(
        typename ToEndpoint::Array const &array) const {
    auto const rawFromData = _toEndpoint.dataFromArray(array);
    auto rawToData = _frameSet->tranInverse(rawFromData);
    return _fromEndpoint.arrayFromData(rawToData);
}

template <class FromEndpoint, class ToEndpoint>
Transform<ToEndpoint, FromEndpoint> Transform<FromEndpoint, ToEndpoint>::getInverse() const {
    auto inverse = std::dynamic_pointer_cast<ast::FrameSet>(_frameSet->getInverse());
    if (!inverse) {
        // don't throw std::bad_cast because it doesn't let you provide debugging info
        std::ostringstream buffer;
        buffer << "FrameSet.getInverse() does not return a FrameSet. Called from: " << _frameSet;
        throw LSST_EXCEPT(pex::exceptions::LogicError, buffer.str());
    }
    return Transform<ToEndpoint, FromEndpoint>(*inverse);
}

template <class FromEndpoint, class ToEndpoint>
Eigen::MatrixXd Transform<FromEndpoint, ToEndpoint>::getJacobian(FromPoint const &x) const {
    try {
        int const nIn = _fromEndpoint.getNAxes();
        int const nOut = _toEndpoint.getNAxes();
        std::vector<double> const point = _fromEndpoint.dataFromPoint(x);

        Eigen::MatrixXd jacobian(nOut, nIn);
        for (int i = 0; i < nOut; ++i) {
            for (int j = 0; j < nIn; ++j) {
                jacobian(i, j) = _frameSet->rate(point, i + 1, j + 1);
            }
        }
        return jacobian;
    } catch (std::bad_alloc const &e) {
        std::throw_with_nested(LSST_EXCEPT(pex::exceptions::MemoryError, "Could not allocate Jacobian."));
    }
}

template <class FromEndpoint, class ToEndpoint>
template <class FirstFromEndpoint>
Transform<FirstFromEndpoint, ToEndpoint> Transform<FromEndpoint, ToEndpoint>::of(
        Transform<FirstFromEndpoint, FromEndpoint> const &first) const {
    if (_fromEndpoint.getNAxes() == first.getToEndpoint().getNAxes()) {
        return Transform<FirstFromEndpoint, ToEndpoint>(*ast::prepend(*_frameSet, *(first.getFrameSet())));
    } else {
        auto message = "Cannot match " + std::to_string(first.getToEndpoint().getNAxes()) +
                       "-D to-endpoint to " + std::to_string(_fromEndpoint.getNAxes()) + "-D from-endpoint.";
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, message);
    }
}

template <class FromEndpoint, class ToEndpoint>
std::ostream &operator<<(std::ostream &os, Transform<FromEndpoint, ToEndpoint> const &transform) {
    auto const frameSet = transform.getFrameSet();
    os << "Transform<" << transform.getFromEndpoint() << ", " << transform.getToEndpoint() << ">";
    return os;
};

#define INSTANTIATE_OVERLOADS(FromEndpoint, ToEndpoint, ExtraEndpoint)                                    \
    template Transform<ExtraEndpoint, ToEndpoint> Transform<FromEndpoint, ToEndpoint>::of<ExtraEndpoint>( \
            Transform<ExtraEndpoint, FromEndpoint> const &) const;
#define INSTANTIATE_TRANSFORM(FromEndpoint, ToEndpoint)                  \
    template class Transform<FromEndpoint, ToEndpoint>;                  \
    INSTANTIATE_OVERLOADS(FromEndpoint, ToEndpoint, GenericEndpoint)     \
    INSTANTIATE_OVERLOADS(FromEndpoint, ToEndpoint, Point2Endpoint)      \
    INSTANTIATE_OVERLOADS(FromEndpoint, ToEndpoint, SpherePointEndpoint) \
    template std::ostream &operator<<<FromEndpoint, ToEndpoint>(         \
            std::ostream &os, Transform<FromEndpoint, ToEndpoint> const &transform);

// explicit instantiations
INSTANTIATE_TRANSFORM(GenericEndpoint, GenericEndpoint);
INSTANTIATE_TRANSFORM(GenericEndpoint, Point2Endpoint);
INSTANTIATE_TRANSFORM(GenericEndpoint, SpherePointEndpoint);
INSTANTIATE_TRANSFORM(Point2Endpoint, GenericEndpoint);
INSTANTIATE_TRANSFORM(Point2Endpoint, Point2Endpoint);
INSTANTIATE_TRANSFORM(Point2Endpoint, SpherePointEndpoint);
INSTANTIATE_TRANSFORM(SpherePointEndpoint, GenericEndpoint);
INSTANTIATE_TRANSFORM(SpherePointEndpoint, Point2Endpoint);
INSTANTIATE_TRANSFORM(SpherePointEndpoint, SpherePointEndpoint);

}  // geom
}  // afw
}  // lsst
