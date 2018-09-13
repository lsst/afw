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

/**
 * A collection of Detectors plus additional coordinate system support.
 *
 * Camera.transform transforms points from one camera coordinate system to another.
 * Camera.getTransform returns a transform between camera coordinate systems.
 * Camera.findDetectors finds all detectors overlapping a specified point.
 */
class Camera : public DetectorCollection {
public:
    using DetectorList = DetectorCollection::List;

    /**
     * Construct a camera
     *
     * @param[in] name  name of camera
     * @param[in] detectorList  a DetectorList in index order
     * @param[in] transformMap  a TransformMap that at least supports FOCAL_PLANE and FIELD_ANGLE coordinates
     * @param[in] pupilFactoryName name of a PupilFactory class for this camera
     */
    Camera(std::string const &name, DetectorList const &detectorList,
           std::shared_ptr<TransformMap> transformMap, std::string const &pupilFactoryName);

    Camera(Camera const &) = delete;
    Camera(Camera &&) = delete;

    Camera & operator=(Camera const &) = delete;
    Camera & operator=(Camera &&) = delete;

    virtual ~Camera() noexcept;

    std::string getName() const { return _name; }

    std::string getPupilFactoryName() const { return _pupilFactoryName; }

    /**
     * Find the detectors that cover a point in any camera system
     *
     * @param[in] point  position to use in lookup (lsst::geom::Point2D)
     * @param[in] cameraSys  camera coordinate system of `point`
     * @returns a list of zero or more Detectors that overlap the specified point
     */
    DetectorList findDetectors(lsst::geom::Point2D const &point, CameraSys const &cameraSys) const;

    /**
     * Find the detectors that cover a list of points in any camera system
     *
     * @param[in] pointList  a list of points (lsst::geom::Point2D)
     * @param[in] cameraSys the camera coordinate system of the points in `pointList`
     * @returns a list of lists; each list contains the names of all detectors
     *    which contain the corresponding point
     */
    std::vector<DetectorList> findDetectorsList(std::vector<lsst::geom::Point2D> const &pointList,
                                   CameraSys const &cameraSys) const;

    /**
     * Get a transform from one CameraSys to another
     *
     * @param[in] fromSys  From CameraSys
     * @param[in] toSys  To CameraSys
     * @returns an afw::geom::TransformPoint2ToPoint2 that transforms from
     *    `fromSys` to `toSys` in the forward direction
     *
     * @throws lsst::pex::exceptions::InvalidParameter if no transform is available.  This includes the case that
     *    `fromSys` specifies a known detector and `toSys` specifies any other detector (known or unknown)
     * @throws KeyError if an unknown detector is specified
     */
    std::shared_ptr<afw::geom::TransformPoint2ToPoint2> getTransform(CameraSys const &fromSys,
                                                                     CameraSys const &toSys) const;

    /**
     * Obtain the transform registry.
     *
     * @returns _transformMap a TransformMap
     *
     * @note _transformMap is immutable, so this should be safe.
     */
    std::shared_ptr<TransformMap const> getTransformMap() const noexcept { return _transformMap; }

    /**
     * Transform a point from one camera coordinate system to another
     *
     * @param[in] point  an lsst::geom::Point2d
     * @param[in] fromSys  transform from this CameraSys
     * @param[in] toSys  transform to this CameraSys
     * @returns point transformed to `toSys` (an lsst::geom::Point2D)
     */
    lsst::geom::Point2D transform(lsst::geom::Point2D const &point, CameraSys const &fromSys,
                                  CameraSys const &toSys) const;

    /**
     * Transform a vector of points from one camera coordinate system to another
     *
     * @param[in] points  an vector of lsst::geom::Point2d
     * @param[in] fromSys  transform from this CameraSys
     * @param[in] toSys  transform to this CameraSys
     * @returns points transformed to `toSys` (a vector of lsst::geom::Point2D)
     */
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
