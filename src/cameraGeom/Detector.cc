/// -*- lsst-c++ -*-
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
#include <sstream>
#include <utility>
#include <unordered_set>

#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"
#include "lsst/afw/table/aggregates.h"
#include "lsst/afw/cameraGeom/Detector.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

namespace {

template <typename Iter>
Iter findAmpIterByName(Iter first, Iter last, std::string const & name) {
    // Given that the number of amplifiers is generally quite small (even LSST
    // only has 16), we just do a linear search in the vector, as adding a map
    // of some kind is definitely more storage and code complexity, without
    // necessarily being any faster.
    auto iter = std::find_if(first, last, [&name](auto const & ptr) { return ptr->getName() == name; });
    if (iter == last) {
        throw LSST_EXCEPT(
            pex::exceptions::InvalidParameterError,
            (boost::format("Amplifier with name %s not found.") % name).str()
        );
    }
    return iter;
}

} // anonymous

std::shared_ptr<Detector::PartialRebuilder> Detector::rebuild() const {
    return std::make_shared<PartialRebuilder>(*this);
}

std::vector<lsst::geom::Point2D> Detector::getCorners(CameraSys const &cameraSys) const {
    std::vector<lsst::geom::Point2D> nativeCorners = lsst::geom::Box2D(getBBox()).getCorners();
    auto nativeToCameraSys = _transformMap->getTransform(getNativeCoordSys(), cameraSys);
    return nativeToCameraSys->applyForward(nativeCorners);
}

std::vector<lsst::geom::Point2D> Detector::getCorners(CameraSysPrefix const &cameraSysPrefix) const {
    return getCorners(makeCameraSys(cameraSysPrefix));
}

lsst::geom::Point2D Detector::getCenter(CameraSys const &cameraSys) const {
    auto ctrPix = lsst::geom::Box2D(getBBox()).getCenter();
    auto transform = getTransform(PIXELS, cameraSys);
    return transform->applyForward(ctrPix);
}

lsst::geom::Point2D Detector::getCenter(CameraSysPrefix const &cameraSysPrefix) const {
    return getCenter(makeCameraSys(cameraSysPrefix));
}

bool Detector::hasTransform(CameraSys const &cameraSys) const { return _transformMap->contains(cameraSys); }

bool Detector::hasTransform(CameraSysPrefix const &cameraSysPrefix) const {
    return hasTransform(makeCameraSys(cameraSysPrefix));
}

template <typename FromSysT, typename ToSysT>
std::shared_ptr<geom::TransformPoint2ToPoint2> Detector::getTransform(FromSysT const &fromSys,
                                                                      ToSysT const &toSys) const {
    return _transformMap->getTransform(makeCameraSys(fromSys), makeCameraSys(toSys));
}

template <typename FromSysT, typename ToSysT>
lsst::geom::Point2D Detector::transform(lsst::geom::Point2D const &point, FromSysT const &fromSys,
                                        ToSysT const &toSys) const {
    return _transformMap->transform(point, makeCameraSys(fromSys), makeCameraSys(toSys));
}

template <typename FromSysT, typename ToSysT>
std::vector<lsst::geom::Point2D> Detector::transform(std::vector<lsst::geom::Point2D> const &points,
                                                     FromSysT const &fromSys, ToSysT const &toSys) const {
    return _transformMap->transform(points, makeCameraSys(fromSys), makeCameraSys(toSys));
}

std::shared_ptr<Amplifier const> Detector::operator[](std::string const &name) const {
    return *findAmpIterByName(_amplifiers.begin(), _amplifiers.end(), name);
}

Detector::Detector(Fields fields, std::shared_ptr<TransformMap const> transformMap,
                   std::vector<std::shared_ptr<Amplifier const>> &&amplifiers) :
    _fields(std::move(fields)), _transformMap(std::move(transformMap)), _amplifiers(std::move(amplifiers))
{
    // Make sure amplifier names are unique.
    std::unordered_set<std::string> amplifierNames;
    for (auto const &ptr : _amplifiers) {
        if (!amplifierNames.insert(ptr->getName()).second) {
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                              (boost::format("Multiple amplifiers with name %s") % ptr->getName()).str());
        }
    }
    // Ensure crosstalk coefficients matrix has the right shape.
    if (hasCrosstalk()) {
        auto crosstalk = getCrosstalk();
        auto shape = crosstalk.getShape();
        assert(shape.size() == 2);  // we've declared this as a 2D array
        if (shape[0] != shape[1]) {
            std::ostringstream os;
            os << "Non-square crosstalk matrix: " << crosstalk << " for detector \"" << getName() << "\"";
            throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
        }
        if (shape[0] != _amplifiers.size()) {
            std::ostringstream os;
            os << "Wrong size crosstalk matrix: " << crosstalk << " for detector \"" << getName() << "\"";
            throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
        }
    }
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
    table::Key<int> id;
    table::Key<int> type;
    table::Key<std::string> serial;
    table::Box2IKey bbox;
    table::Point2DKey pixelSize;
    table::Point2DKey fpPosition;
    table::Point2DKey refPoint;
    table::Key<lsst::geom::Angle> yaw;
    table::Key<lsst::geom::Angle> pitch;
    table::Key<lsst::geom::Angle> roll;
    table::Key<int> transformMap;
    table::Key<table::Array<float>> crosstalk;
    table::Key<std::string> physicalType;

    PersistenceHelper(table::Schema const & existing) :
        schema(existing),
        name(schema["name"]),
        id(schema["id"]),
        type(schema["type"]),
        serial(schema["serial"]),
        bbox(schema["bbox"]),
        pixelSize(schema["pixelSize"]),
        fpPosition(schema["fpPosition"]),
        refPoint(schema["refPoint"]),
        yaw(schema["yaw"]),
        pitch(schema["pitch"]),
        roll(schema["roll"]),
        transformMap(schema["transformMap"]),
        crosstalk(schema["crosstalk"])
    {
        auto setKeyIfPresent = [this](auto & key, std::string const & name) {
            try {
                key = schema[name];
            } catch (pex::exceptions::NotFoundError &) {}
        };
        // This field was not part of the original Detector minimal
        // schema, but needed to be added
        setKeyIfPresent(physicalType, "physicalType");
    }

private:

    PersistenceHelper() :
        schema(),
        name(schema.addField<std::string>("name", "Name of the detector", "", 0)),
        id(schema.addField<int>("id", "Integer ID for the detector", "")),
        type(schema.addField<int>("type", "Raw DetectorType enum value", "")),
        serial(schema.addField<std::string>("serial", "Serial name of the detector", "", 0)),
        bbox(table::Box2IKey::addFields(schema, "bbox", "Detector bounding box", "pixel")),
        pixelSize(table::Point2DKey::addFields(schema, "pixelSize", "Physical pixel size", "mm")),
        fpPosition(table::Point2DKey::addFields(schema, "fpPosition",
                                                "Focal plane position of reference point", "mm")),
        refPoint(table::Point2DKey::addFields(schema, "refPoint",
                                              "Pixel position of reference point", "pixel")),
        yaw(schema.addField<lsst::geom::Angle>("yaw", "Rotation about Z (X to Y), 1st rotation")),
        pitch(schema.addField<lsst::geom::Angle>("pitch", "Rotation about Y' (Z'=Z to X'), 2nd rotation")),
        roll(schema.addField<lsst::geom::Angle>("roll", "Rotation about X'' (Y''=Y' to Z''), 3rd rotation")),
        transformMap(schema.addField<int>("transformMap", "Archive ID of TransformMap", "")),
        crosstalk(schema.addField<table::Array<float>>("crosstalk", "Crosstalk matrix, flattened", "", 0)),
        physicalType(schema.addField<std::string>("physicalType", "Physical type of the detector", "", 0))
    {}

    PersistenceHelper(PersistenceHelper const &) = delete;
    PersistenceHelper(PersistenceHelper &&) = delete;

    PersistenceHelper & operator=(PersistenceHelper const &) = delete;
    PersistenceHelper & operator=(PersistenceHelper &&) = delete;

};

} // anonymous

class Detector::Factory : public table::io::PersistableFactory {
public:

    Factory() : PersistableFactory("Detector") {}

    std::shared_ptr<table::io::Persistable> read(InputArchive const& archive,
                                                 CatalogVector const& catalogs) const override {
        // N.b. can't use "auto const keys" as cctor is deleted
        auto const & keys = PersistenceHelper(catalogs.front().getSchema());

        LSST_ARCHIVE_ASSERT(catalogs.size() == 2u);
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        auto const & record = catalogs.front().front();

        std::vector<std::shared_ptr<Amplifier const>> amps;
        amps.reserve(catalogs.back().size());
        for (auto const & record : catalogs.back()) {
            amps.push_back(Amplifier::Builder::fromRecord(record).finish());
        }

        auto flattenedMatrix = record.get(keys.crosstalk);
        ndarray::Array<float, 2, 2> crosstalk;
        if (!flattenedMatrix.isEmpty()) {
            crosstalk = ndarray::allocate(amps.size(), amps.size());
            ndarray::flatten<1>(crosstalk) = flattenedMatrix;
        }

        // get values for not-always-present fields if present
        const auto physicalType = keys.physicalType.isValid() ? record.get(keys.physicalType) : "";
        Fields fields = {
            record.get(keys.name),
            record.get(keys.id),
            static_cast<DetectorType>(record.get(keys.type)),
            record.get(keys.serial),
            record.get(keys.bbox),
            Orientation(
                record.get(keys.fpPosition),
                record.get(keys.refPoint),
                record.get(keys.yaw),
                record.get(keys.pitch),
                record.get(keys.roll)
            ),
            lsst::geom::Extent2D(record.get(keys.pixelSize)),
            crosstalk,
            physicalType
        };

        return std::shared_ptr<Detector>(
            new Detector(
                std::move(fields),
                archive.get<TransformMap>(record.get(keys.transformMap)),
                std::move(amps)
            )
        );
    }

    static Factory const registration;

};

Detector::Factory const Detector::Factory::registration;

std::string Detector::getPersistenceName() const {
    return "Detector";
}

std::string Detector::getPythonModule() const {
    return "lsst.afw.cameraGeom";
}

void Detector::write(OutputArchiveHandle& handle) const {
    auto const & keys = PersistenceHelper::get();

    auto cat = handle.makeCatalog(keys.schema);
    auto record = cat.addNew();
    record->set(keys.name, getName());
    record->set(keys.id, getId());
    record->set(keys.type, static_cast<int>(getType()));
    record->set(keys.serial, getSerial());
    record->set(keys.bbox, getBBox());
    record->set(keys.pixelSize, lsst::geom::Point2D(getPixelSize()));
    auto orientation = getOrientation();
    record->set(keys.fpPosition, orientation.getFpPosition());
    record->set(keys.refPoint, orientation.getReferencePoint());
    record->set(keys.yaw, orientation.getYaw());
    record->set(keys.pitch, orientation.getPitch());
    record->set(keys.roll, orientation.getRoll());
    record->set(keys.transformMap, handle.put(getTransformMap()));

    auto flattenMatrix = [](ndarray::Array<float const, 2> const & matrix) {
        // copy because the original isn't guaranteed to have
        // row-major contiguous elements
        ndarray::Array<float, 2, 2> copied = ndarray::copy(matrix);
        // make a view to the copy
        ndarray::Array<float, 1, 1> flattened = ndarray::flatten<1>(copied);
        return flattened;
    };

    record->set(keys.crosstalk, flattenMatrix(getCrosstalk()));
    record->set(keys.physicalType, getPhysicalType());
    handle.saveCatalog(cat);

    auto ampCat = handle.makeCatalog(Amplifier::getRecordSchema());
    ampCat.reserve(getAmplifiers().size());
    for (auto const & amp : getAmplifiers()) {
        auto record = ampCat.addNew();
        amp->toRecord(*record);
    }
    handle.saveCatalog(ampCat);
}


std::shared_ptr<Amplifier::Builder> Detector::Builder::operator[](std::string const &name) const {
    return *findAmpIterByName(_amplifiers.begin(), _amplifiers.end(), name);
}

void Detector::Builder::append(std::shared_ptr<Amplifier::Builder> builder) {
    _amplifiers.push_back(std::move(builder));
}

std::vector<std::shared_ptr<Amplifier::Builder>> Detector::Builder::rebuildAmplifiers(
    Detector const & detector
) {
    std::vector<std::shared_ptr<Amplifier::Builder>> result;
    result.reserve(detector.size());
    for (auto const & ampPtr : detector) {
        result.push_back(std::make_shared<Amplifier::Builder>(*ampPtr));
    }
    return result;
}

Detector::Builder::Builder(std::string const & name, int id) {
    _fields.name = name;
    _fields.id = id;
}

std::vector<std::shared_ptr<Amplifier const>> Detector::Builder::finishAmplifiers() const {
    std::vector<std::shared_ptr<Amplifier const>> result;
    result.reserve(_amplifiers.size());
    for (auto const & ampBuilderPtr : _amplifiers) {
        result.push_back(ampBuilderPtr->finish());
    }
    return result;
}


Detector::PartialRebuilder::PartialRebuilder(Detector const & detector) :
    Builder(detector._fields, rebuildAmplifiers(detector)),
    _transformMap(detector.getTransformMap())
{}

std::shared_ptr<Detector const> Detector::PartialRebuilder::finish() const {
    auto amplifiers = finishAmplifiers();
    return std::shared_ptr<Detector>(new Detector(getFields(), _transformMap, std::move(amplifiers)));
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


void Detector::InCameraBuilder::setTransformFromPixelsTo(
    CameraSysPrefix const & toSys,
    std::shared_ptr<afw::geom::TransformPoint2ToPoint2 const> transform
) {
    return setTransformFromPixelsTo(makeCameraSys(toSys), std::move(transform));
}

void Detector::InCameraBuilder::setTransformFromPixelsTo(
    CameraSys const & toSys,
    std::shared_ptr<afw::geom::TransformPoint2ToPoint2 const> transform
) {
    if (toSys.getDetectorName() != getName()) {
        throw LSST_EXCEPT(
            pex::exceptions::InvalidParameterError,
            (boost::format("Cannot add coordinate system for detector '%s' to detector '%s'.") %
             toSys.getDetectorName() % getName()).str()
        );
    }
    auto iter = findConnection(_connections.begin(), _connections.end(), toSys);
    if (iter == _connections.end()) {
        _connections.push_back(
            TransformMap::Connection{transform, getNativeCoordSys(), toSys}
        );
    } else {
        iter->transform = transform;
    }
}

bool Detector::InCameraBuilder::discardTransformFromPixelsTo(CameraSysPrefix const & toSys) {
    return discardTransformFromPixelsTo(makeCameraSys(toSys));
}

bool Detector::InCameraBuilder::discardTransformFromPixelsTo(CameraSys const & toSys) {
    if (toSys.getDetectorName() != getName()) {
        throw LSST_EXCEPT(
            pex::exceptions::InvalidParameterError,
            (boost::format("Cannot add coordinate system for detector '%s' to detector '%s'.") %
             toSys.getDetectorName() % getName()).str()
        );
    }
    auto iter = findConnection(_connections.begin(), _connections.end(), toSys);
    if (iter != _connections.end()) {
        _connections.erase(iter);
        return true;
    }
    return false;
}


Detector::InCameraBuilder::InCameraBuilder(Detector const & detector) :
    Builder(detector.getFields(), rebuildAmplifiers(detector))
{}

Detector::InCameraBuilder::InCameraBuilder(std::string const & name, int id) :
    Builder(name, id)
{}


std::shared_ptr<Detector const> Detector::InCameraBuilder::finish(
    std::shared_ptr<TransformMap const> transformMap
) const {
    auto amplifiers = finishAmplifiers();
    return std::shared_ptr<Detector const>(
        new Detector(getFields(), std::move(transformMap), std::move(amplifiers))
    );
}


//
// Explicit instantiations
//
#define INSTANTIATE(FROMSYS, TOSYS)                                                                         \
    template std::shared_ptr<geom::TransformPoint2ToPoint2> Detector::getTransform(FROMSYS const &,         \
                                                                                   TOSYS const &) const;    \
    template lsst::geom::Point2D Detector::transform(lsst::geom::Point2D const &, FROMSYS const &,          \
                                                     TOSYS const &) const;                                  \
    template std::vector<lsst::geom::Point2D> Detector::transform(std::vector<lsst::geom::Point2D> const &, \
                                                                  FROMSYS const &, TOSYS const &) const;

INSTANTIATE(CameraSys, CameraSys);
INSTANTIATE(CameraSys, CameraSysPrefix);
INSTANTIATE(CameraSysPrefix, CameraSys);
INSTANTIATE(CameraSysPrefix, CameraSysPrefix);

}  // namespace cameraGeom

namespace table {
namespace io {

template class PersistableFacade<cameraGeom::Detector>;

}  // namespace io
}  // namespace table


}  // namespace afw
}  // namespace lsst
