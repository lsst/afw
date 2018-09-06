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

#ifndef LSST_AFW_CAMERAGEOM_CAMERA_H
#define LSST_AFW_CAMERAGEOM_CAMERA_H

#include <string>
#include <memory>

#include "lsst/afw/cameraGeom/DetectorCollection.h"
#include "lsst/afw/cameraGeom/TransformMap.h"
#include "lsst/afw/cameraGeom/CameraSys.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

class Camera : public DetectorCollection {


public:
    using DetectorList = DetectorCollection::List;

    Camera(std::string const &name, DetectorList const &detectorList,
           std::shared_ptr<TransformMap> transformMap, std::string const &pupilFactoryName);

    Camera(Camera const &) = delete;
    Camera(Camera &&) = delete;

    Camera & operator=(Camera const &) = delete;
    Camera & operator=(Camera &&) = delete;

    virtual ~Camera() noexcept;

    std::string getName() const { return _name; }

    std::string getPupilFactoryName() const { return _pupilFactoryName; }

    DetectorList findDetectors(lsst::geom::Point2D const &point, CameraSys const &cameraSys) const;

    std::vector<DetectorList> findDetectorsList(std::vector<lsst::geom::Point2D> const &pointList,
                                   CameraSys const &cameraSys) const;

    std::shared_ptr<afw::geom::TransformPoint2ToPoint2> getTransform(CameraSys const &fromSys,
                                                                     CameraSys const &toSys) const;

    std::shared_ptr<TransformMap const> getTransformMap() const noexcept { return _transformMap; }

    lsst::geom::Point2D transform(lsst::geom::Point2D const &point, CameraSys const &fromSys,
                                  CameraSys const &toSys) const;

    std::vector<lsst::geom::Point2D> transform(std::vector<lsst::geom::Point2D> const &points, 
                                               CameraSys const &fromSys,
                                               CameraSys const &toSys) const;

private:

    std::string _name;
    std::shared_ptr<TransformMap const> _transformMap;
    std::string _pupilFactoryName;


};

} // namespace cameraGeom
} // namespace afw
} // namespace lsst


#endif // LSST_AFW_CAMERAGEOM_CAMERA_H
