/*
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "lsst/afw/cameraGeom/Camera.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

namespace {
    // Set this as a function to ensure FOCAL_PLANE is defined before use.
    CameraSys const getNativeCameraSys() { return FOCAL_PLANE; }

std::shared_ptr<afw::geom::TransformPoint2ToPoint2> getTransformFromOneTransformMap(
    Camera const &camera, CameraSys const &fromSys, CameraSys const &toSys) {

    if (fromSys.hasDetectorName()) {
        auto det = camera[fromSys.getDetectorName()];
        return det->getTransformMap().getTransform(fromSys, toSys);
    } else if (toSys.hasDetectorName()) {
        auto det = camera[toSys.getDetectorName()];
        return det->getTransformMap().getTransform(fromSys, toSys);
    } else {
        return camera.getTransformMap()->getTransform(fromSys, toSys);
    }
}

} // anonymous

Camera::Camera(std::string const &name, DetectorList const &detectorList,
               std::shared_ptr<TransformMap> transformMap, std::string const &pupilFactoryName) :
    DetectorCollection(detectorList),
    _name(name), _transformMap(std::move(transformMap)), _pupilFactoryName(pupilFactoryName)
    {}

Camera::~Camera() noexcept = default;

Camera::DetectorList Camera::findDetectors(lsst::geom::Point2D const &point, CameraSys const &cameraSys) const {
    auto transform = getTransformFromOneTransformMap(*this, cameraSys, getNativeCameraSys());
    auto nativePoint = transform->applyForward(point);

    DetectorList detectorList;
    for (auto const &item: getIdMap()) {
        auto detector = item.second;
        auto nativeToPixels = detector->getTransform(getNativeCameraSys(), PIXELS);
        auto pointPixels = nativeToPixels->applyForward(nativePoint);
        if (lsst::geom::Box2D(detector->getBBox()).contains(pointPixels)) {
            detectorList.push_back(std::move(detector));
        }
    }
    return detectorList;
}

std::vector<Camera::DetectorList> Camera::findDetectorsList(std::vector<lsst::geom::Point2D> const &pointList,
                                                            CameraSys const &cameraSys) const {
    auto transform = getTransformFromOneTransformMap(*this, cameraSys, getNativeCameraSys());
    std::vector<DetectorList> detectorListList(pointList.size());

    auto nativePointList = transform->applyForward(pointList);

    for (auto const &item: getIdMap()) {
        auto const &detector = item.second;
        auto nativeToPixels = detector->getTransform(getNativeCameraSys(), PIXELS);
        auto pointPixelsList = nativeToPixels->applyForward(nativePointList);
        for (std::size_t i = 0; i < pointPixelsList.size(); ++i) {
            auto const &pointPixels = pointPixelsList[i];
            if (lsst::geom::Box2D(detector->getBBox()).contains(pointPixels)) {
                detectorListList[i].push_back(detector);
            }
        }
    }
    return detectorListList;
}

std::shared_ptr<afw::geom::TransformPoint2ToPoint2> Camera::getTransform(CameraSys const &fromSys,
                                                                         CameraSys const &toSys) const {
    auto fromSysToNative = getTransformFromOneTransformMap(*this, fromSys, getNativeCameraSys());
    auto nativeToToSys = getTransformFromOneTransformMap(*this, getNativeCameraSys(), toSys);
    return fromSysToNative->then(*nativeToToSys);
}

lsst::geom::Point2D Camera::transform(lsst::geom::Point2D const &point, CameraSys const &fromSys,
                                      CameraSys const &toSys) const {
    auto transform = getTransform(fromSys, toSys);
    return transform->applyForward(point);
}
std::vector<lsst::geom::Point2D> Camera::transform(std::vector<lsst::geom::Point2D> const &points,
                                                   CameraSys const &fromSys,
                                                   CameraSys const &toSys) const {
    auto transform = getTransform(fromSys, toSys);
    return transform->applyForward(points);
}






} // namespace cameraGeom
} // namespace afw
} // namespace lsst

