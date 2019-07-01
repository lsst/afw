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

#include "lsst/afw/table/io/Persistable.cc"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/cameraGeom/Camera.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

namespace {

// Set this as a function to ensure FOCAL_PLANE is defined before use.
CameraSys const getNativeCameraSys() { return FOCAL_PLANE; }

/**
 * Get a transform from one TransformMap
 *
 * `fromSys` and `toSys` must both be present in the same TransformMap, but that TransformMap may be from
 *    any detector or this camera object.
 *
 * @param[in] fromSys  Camera coordinate system of input points
 * @param[in] toSys  Camera coordinate system of returned points
 * @returns an afw::geom::TransformPoint2ToPoint2 that transforms from `fromSys` to `toSys` in the forward
 *    direction
 *
 * @throws lsst::pex::exceptions::InvalidParameter if no transform is available.  This includes the case that
 *    `fromSys` specifies a known detector and `toSys` specifies any other detector (known or unknown)
 * @throws KeyError if an unknown detector is specified
 */
std::shared_ptr<afw::geom::TransformPoint2ToPoint2> getTransformFromOneTransformMap(
    Camera const &camera, CameraSys const &fromSys, CameraSys const &toSys) {

    if (fromSys.hasDetectorName()) {
        auto det = camera[fromSys.getDetectorName()];
        return det->getTransformMap()->getTransform(fromSys, toSys);
    } else if (toSys.hasDetectorName()) {
        auto det = camera[toSys.getDetectorName()];
        return det->getTransformMap()->getTransform(fromSys, toSys);
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

Camera::DetectorList Camera::findDetectors(lsst::geom::Point2D const &point,
                                           CameraSys const &cameraSys) const {
    auto transform = getTransformFromOneTransformMap(*this, cameraSys, getNativeCameraSys());
    auto nativePoint = transform->applyForward(point);

    DetectorList detectorList;
    for (auto const &item : getIdMap()) {
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
    try {
        return getTransformMap()->getTransform(fromSys, toSys);
    } catch (pex::exceptions::InvalidParameterError &) {}

    // If the Camera was constructed after DM-14980 using the makeCamera*
    // methods in cameraFactory.py, the Camera and all Detectors share a
    // single TransformMap that knows about all of the coordinate systems. In
    // that case the above call should succeed (unless the requested
    // coordinate systems are totally bogus).
    //
    // But if someone built this Camera by hand, the Detectors will know about
    // only the coordinate systems associated with them, while the Camera
    // itself only knows about coordinate systems that aren't associated with
    // any particular Detector.  In that case we need to (in general) look up
    // transforms in multiple places and connect them using the "native camera
    // sys" that's known to everything (at least usually FOCAL_PLANE).
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


namespace {

class PersistenceHelper {
public:

    static PersistenceHelper const & get() {
        static PersistenceHelper const instance;
        return instance;
    }

    table::Schema schema;
    table::Key<std::string> name;
    table::Key<std::string> pupilFactoryName;
    table::Key<int> transformMap;

private:

    PersistenceHelper() :
        schema(),
        name(schema.addField<std::string>("name", "Camera name", "", 0)),
        pupilFactoryName(schema.addField<std::string>("pupilFactoryName",
                                                      "Fully-qualified name of a Python PupilFactory class",
                                                      "", 0)),
        transformMap(schema.addField<int>("transformMap", "archive ID for Camera's TransformMap"))
    {
    }

    PersistenceHelper(PersistenceHelper const &) = delete;
    PersistenceHelper(PersistenceHelper &&) = delete;

    PersistenceHelper & operator=(PersistenceHelper const &) = delete;
    PersistenceHelper & operator=(PersistenceHelper &&) = delete;

};

} // anonymous


class Camera::Factory : public table::io::PersistableFactory {
public:

    Factory() : table::io::PersistableFactory("Camera") {}

    std::shared_ptr<Persistable> read(InputArchive const& archive,
                                      CatalogVector const& catalogs) const override {
         // can't use make_shared because ctor is protected
        return std::shared_ptr<Camera>(new Camera(archive, catalogs));
    }

    static Factory const registration;

};

Camera::Factory const Camera::Factory::registration;


Camera::Camera(table::io::InputArchive const & archive, table::io::CatalogVector const & catalogs) :
    DetectorCollection(archive, catalogs)
    // deferred initalization for data members is not ideal, but better than
    // trying to initialize them before validating the archive
{
    auto const & keys = PersistenceHelper::get();
    LSST_ARCHIVE_ASSERT(catalogs.size() >= 2u);
    auto const & cat = catalogs[1];
    LSST_ARCHIVE_ASSERT(cat.getSchema() == keys.schema);
    LSST_ARCHIVE_ASSERT(cat.size() == 1u);
    auto const & record = cat.front();
    _name = record.get(keys.name);
    _pupilFactoryName = record.get(keys.pupilFactoryName);
    _transformMap = archive.get<TransformMap>(record.get(keys.transformMap));
}


std::string Camera::getPersistenceName() const { return "Camera"; }

void Camera::write(OutputArchiveHandle& handle) const {
    DetectorCollection::write(handle);
    auto const & keys = PersistenceHelper::get();
    auto cat = handle.makeCatalog(keys.schema);
    auto record = cat.addNew();
    record->set(keys.name, getName());
    record->set(keys.pupilFactoryName, getPupilFactoryName());
    record->set(keys.transformMap, handle.put(getTransformMap()));
    handle.saveCatalog(cat);
}

} // namespace cameraGeom

namespace table {
namespace io {

template class PersistableFacade<cameraGeom::Camera>;

} // namespace io
} // namespace table

} // namespace afw
} // namespace lsst

