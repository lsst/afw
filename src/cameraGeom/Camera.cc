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

} // anonymoous

Camera::~Camera() noexcept = default;

Camera::Builder Camera::rebuild() const {
    return Camera::Builder(*this);
}

Camera::DetectorList Camera::findDetectors(lsst::geom::Point2D const &point,
                                           CameraSys const &cameraSys) const {
    auto nativePoint = transform(point, cameraSys, getNativeCameraSys());

    DetectorList detectorList;
    for (auto const &item : getIdMap()) {
        auto detector = item.second;
        auto pointPixels = detector->transform(nativePoint, getNativeCameraSys(), PIXELS);
        if (lsst::geom::Box2D(detector->getBBox()).contains(pointPixels)) {
            detectorList.push_back(std::move(detector));
        }
    }
    return detectorList;
}

std::vector<Camera::DetectorList> Camera::findDetectorsList(std::vector<lsst::geom::Point2D> const &pointList,
                                                            CameraSys const &cameraSys) const {
    std::vector<DetectorList> detectorListList(pointList.size());
    auto nativePointList = transform(pointList, cameraSys, getNativeCameraSys());

    for (auto const &item: getIdMap()) {
        auto const &detector = item.second;
        auto pointPixelsList = detector->transform(nativePointList, getNativeCameraSys(), PIXELS);
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
    return getTransformMap()->getTransform(fromSys, toSys);
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

std::shared_ptr<Detector::InCameraBuilder> Camera::makeDetectorBuilder(std::string const & name, int id) {
    return std::shared_ptr<Detector::InCameraBuilder>(new Detector::InCameraBuilder(name, id));
}

std::shared_ptr<Detector::InCameraBuilder> Camera::makeDetectorBuilder(Detector const & detector) {
    return std::shared_ptr<Detector::InCameraBuilder>(new Detector::InCameraBuilder(detector));
}


std::vector<TransformMap::Connection> const & Camera::getDetectorBuilderConnections(
    Detector::InCameraBuilder const & detector
) {
    return detector._connections;
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

Camera::Camera(std::string const & name, DetectorList const & detectors,
               std::shared_ptr<TransformMap const> transformMap, std::string const & pupilFactoryName) :
    DetectorCollection(detectors),
    _name(name),
    _pupilFactoryName(pupilFactoryName),
    _transformMap(transformMap)
{}

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


Camera::Builder::Builder(std::string const & name) : _name(name) {}

Camera::Builder::Builder(Camera const & camera) :
    _name(camera.getName()),
    _pupilFactoryName(camera.getPupilFactoryName())
{
    // Add Detector Builders for all Detectors; does not (yet) include
    // coordinate transform information.
    for (auto const & pair : camera.getIdMap()) {
        BaseCollection::add(Camera::makeDetectorBuilder(*pair.second));
    }
    // Iterate over connections in TransformMap, distributing them between the
    // Camera Builder and the Detector Builders.
    for (auto const & connection : camera.getTransformMap()->getConnections()) {
        // asserts below are on Detector, Camera, and TransformMap invariants:
        //  - Connections should always be from native sys to something else.
        //  - The only connections between full-camera and per-detector sys
        //    should be from the camera native sys (FOCAL_PLANE) to the
        //    detector native sys (PIXELS).
        //  - When TransformMap standardizes connections, it should maintain
        //    these directions, as that's consistent with "pointing away" from
        //    the overall reference sys (the camera native sys).
        if (connection.fromSys.hasDetectorName()) {
            assert(connection.toSys.getDetectorName() == connection.fromSys.getDetectorName());
            auto detector = (*this)[connection.fromSys.getDetectorName()];
            assert(connection.fromSys == detector->getNativeCoordSys());
            detector->setTransformFromPixelsTo(CameraSysPrefix(connection.toSys.getSysName()),
                                               connection.transform);
        } else {
            assert(connection.fromSys == getNativeCameraSys());
            if (!connection.toSys.hasDetectorName()) {
                _connections.push_back(connection);
            }
            // We ignore the FOCAL_PLANE to PIXELS transforms transforms, as
            // those are always regenerated from the Orientation when we
            // rebuild the Camera.
        }
    }
}

std::shared_ptr<Camera const> Camera::Builder::finish() const {
    // Make a big vector of all coordinate transform connections;
    // start with general transforms for the camera as a whole:
    std::vector<TransformMap::Connection> connections(_connections);
    // Loop over detectors and add the transforms from FOCAL_PLANE
    // to PIXELS (via the Orientation), and then any extra transforms
    // from PIXELS to other things.
    for (auto const & pair : getIdMap()) {
        auto const & detectorBuilder = *pair.second;
        connections.push_back(
            TransformMap::Connection{
                detectorBuilder.getOrientation().makeFpPixelTransform(detectorBuilder.getPixelSize()),
                getNativeCameraSys(),
                detectorBuilder.getNativeCoordSys()
            }
        );
        connections.insert(connections.end(),
                           getDetectorBuilderConnections(detectorBuilder).begin(),
                           getDetectorBuilderConnections(detectorBuilder).end());
    }
    // Make a single big TransformMap.
    auto transformMap = TransformMap::make(getNativeCameraSys(), connections);
    // Make actual Detector objects, giving each the full TransformMap.
    DetectorList detectors;
    detectors.reserve(size());
    for (auto const & pair : getIdMap()) {
        auto const & detectorBuilder = *pair.second;
        detectors.push_back(detectorBuilder.finish(transformMap));
    }
    return std::shared_ptr<Camera>(new Camera(_name, detectors, std::move(transformMap), _pupilFactoryName));
}


namespace {

template <typename Iter>
Iter findConnection(Iter first, Iter last, CameraSys const & toSys) {
    return std::find_if(
        first, last,
        [&toSys](auto const & connection) {
            return connection.toSys == toSys;
        }
    );
}

} // anonymous

void Camera::Builder::setTransformFromFocalPlaneTo(
    CameraSys const & toSys,
    std::shared_ptr<afw::geom::TransformPoint2ToPoint2 const> transform
) {
    if (toSys.hasDetectorName()) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicError,
            (boost::format("%s should be added to Detector %s, not Camera") %
             toSys.getSysName() % toSys.getDetectorName()).str()
        );
    }
    auto iter = findConnection(_connections.begin(), _connections.end(), toSys);
    if (iter == _connections.end()) {
        _connections.push_back(
            TransformMap::Connection{transform, getNativeCameraSys(), toSys}
        );
    } else {
        iter->transform = transform;
    }
}

bool Camera::Builder::discardTransformFromFocalPlaneTo(CameraSys const & toSys) {
    auto iter = findConnection(_connections.begin(), _connections.end(), toSys);
    if (iter != _connections.end()) {
        _connections.erase(iter);
        return true;
    }
    return false;
}

std::shared_ptr<Detector::InCameraBuilder> Camera::Builder::add(std::string const & name, int id) {
    auto detector = makeDetectorBuilder(name, id);
    BaseCollection::add(detector);
    return detector;
}

} // namespace cameraGeom

namespace table {
namespace io {

template class PersistableFacade<cameraGeom::Camera>;

} // namespace io
} // namespace table

} // namespace afw
} // namespace lsst

