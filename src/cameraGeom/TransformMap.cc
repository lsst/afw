/*
 * LSST Data Management System
 * Copyright 2014 LSST Corporation.
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
#include <sstream>
#include <type_traits>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/cameraGeom/TransformMap.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

namespace {

/*
 * Make a FrameSet from a map of camera system: transform
 *
 * @tparam Map Any type satisfying the STL map API and mapping CameraSys to
 *             shared_ptr<geom::TransformPoint2ToPoint2>.
 *
 * @param reference  Coordinate system from which each Transform in `transforms` converts.
 * @param transforms  A map whose keys are camera coordinate systems, and whose values
 *                    point to Transforms that convert from `reference` to the corresponding key.
 *                    All Transforms must be invertible.
 * @return an ast::FrameSet containing one ast::Frame(2) for `reference` and an ast::Frame(2)
 *      for each Transform in `transforms`, connected by suitable mappings.
 *
 * @throws lsst::pex::exceptions::InvalidParameterError Thrown if `transforms` contains
 *         the `reference` camera system as a key, or if any Transform is not invertible.
 */
template <class Map>
std::unique_ptr<ast::FrameSet> makeTransforms(CameraSys const &reference, Map const &transforms) {
    ast::Frame rootFrame(2, "Ident=" + reference.getSysName());
    auto result = std::unique_ptr<ast::FrameSet>(new ast::FrameSet(rootFrame));

    for (auto const &keyValue : transforms) {
        CameraSys const key = keyValue.first;
        std::shared_ptr<geom::TransformPoint2ToPoint2> const value = keyValue.second;

        if (key == reference) {
            std::ostringstream buffer;
            buffer << "Cannot specify a Transform from " << reference << " to itself.";
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
        }
        if (!value->hasForward()) {
            std::ostringstream buffer;
            buffer << *value << " from " << key << " has no forward transform.";
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
        }
        if (!value->hasInverse()) {
            std::ostringstream buffer;
            buffer << *value << " from " << key << " has no inverse transform.";
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
        }

        auto toFrame = ast::Frame(2, "Ident=" + key.getSysName());
        result->addFrame(ast::FrameSet::BASE, *(value->getMapping()), toFrame);
    }
    return result;
}

/*
 * Make a map of camera system: frame number for a FrameSet constructed by makeTransforms.
 *
 * @tparam Map Any type satisfying the STL map API and mapping CameraSys to
 *             shared_ptr<geom::TransformPoint2ToPoint2>.
 *
 * @param reference  Coordinate system from which each Transform in `transforms` converts.
 * @param transforms  A map whose keys are camera coordinate systems, and whose values
 *                    point to Transforms that convert from `reference` to the corresponding key.
 *                    All Transforms must be invertible.
 * @return a map from `reference` and each key in `transforms` to the corresponding frame number
 *         in the ast::FrameSet returned by `makeTransforms(reference, transforms)`.
 *
 * @warning Does not perform input validation.
 */
template <class Map>
std::unordered_map<CameraSys, int> makeTranslator(CameraSys const &reference, Map const &transforms) {
    std::unordered_map<CameraSys, int> result({std::make_pair(reference, 1)});
    int nFrames = 1;

    for (auto const &keyValue : transforms) {
        CameraSys const key = keyValue.first;
        result.emplace(key, ++nFrames);
    }
    return result;
}

}  // namespace

lsst::afw::geom::Point2Endpoint TransformMap::_pointConverter;

TransformMap::TransformMap(
        CameraSys const &reference,
        std::unordered_map<CameraSys, std::shared_ptr<geom::TransformPoint2ToPoint2>> const &transforms)
        : _transforms(makeTransforms(reference, transforms)),
          _frameIds(makeTranslator(reference, transforms)) {}

// TransformMap is immutable, so we can just copy the shared_ptr
TransformMap::TransformMap(TransformMap const &other) = default;

// Cannot do any move optimizations without breaking immutability
TransformMap::TransformMap(TransformMap const &&other) : TransformMap(other) {}

// All resources owned by value or by smart pointer
TransformMap::~TransformMap() = default;

lsst::geom::Point2D TransformMap::transform(lsst::geom::Point2D const &point, CameraSys const &fromSys,
                                            CameraSys const &toSys) const {
    auto mapping = _getMapping(fromSys, toSys);
    return _pointConverter.pointFromData(mapping->applyForward(_pointConverter.dataFromPoint(point)));
}

std::vector<lsst::geom::Point2D> TransformMap::transform(std::vector<lsst::geom::Point2D> const &pointList,
                                                         CameraSys const &fromSys,
                                                         CameraSys const &toSys) const {
    auto mapping = _getMapping(fromSys, toSys);
    return _pointConverter.arrayFromData(mapping->applyForward(_pointConverter.dataFromArray(pointList)));
}

bool TransformMap::contains(CameraSys const &system) const noexcept { return _frameIds.count(system) > 0; }

std::shared_ptr<geom::TransformPoint2ToPoint2> TransformMap::getTransform(CameraSys const &fromSys,
                                                                          CameraSys const &toSys) const {
    return std::make_shared<geom::TransformPoint2ToPoint2>(*_getMapping(fromSys, toSys));
}

int TransformMap::_getFrame(CameraSys const &system) const {
    try {
        return _frameIds.at(system);
    } catch (std::out_of_range const &e) {
        std::ostringstream buffer;
        buffer << "Unsupported coordinate system: " << system;
        std::throw_with_nested(LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str()));
    }
}

std::shared_ptr<ast::Mapping const> TransformMap::_getMapping(CameraSys const &fromSys,
                                                              CameraSys const &toSys) const {
    return _transforms->getMapping(_getFrame(fromSys), _getFrame(toSys));
}

size_t TransformMap::size() const noexcept { return _frameIds.size(); }
}  // namespace cameraGeom
}  // namespace afw
}  // namespace lsst
