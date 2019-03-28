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

#include <sstream>
#include <utility>

#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"
#include "lsst/afw/table/aggregates.h"
#include "lsst/afw/cameraGeom/Detector.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

Detector::Detector(std::string const &name, int id, DetectorType type, std::string const &serial,
                   lsst::geom::Box2I const &bbox,
                   std::vector<std::shared_ptr<Amplifier const>> const &amplifiers,
                   Orientation const &orientation, lsst::geom::Extent2D const &pixelSize,
                   TransformMap::Transforms const &transforms, CrosstalkMatrix const &crosstalk,
                   std::string const &physicalType) :
    Detector(name, id, type, serial, bbox, amplifiers, orientation, pixelSize,
             TransformMap::make(CameraSys(PIXELS, name), transforms),
             crosstalk, physicalType)
{}

Detector::Detector(std::string const &name, int id, DetectorType type, std::string const &serial,
                   lsst::geom::Box2I const &bbox,
                   std::vector<std::shared_ptr<Amplifier const>> const &amplifiers,
                   Orientation const &orientation, lsst::geom::Extent2D const &pixelSize,
                   std::shared_ptr<TransformMap const> transformMap, CrosstalkMatrix const &crosstalk,
                   std::string const &physicalType) :
    _name(name),
    _id(id),
    _type(type),
    _serial(serial),
    _bbox(bbox),
    _amplifiers(amplifiers),
    _amplifierMap(),
    _orientation(orientation),
    _pixelSize(pixelSize),
    _nativeSys(CameraSys(PIXELS, name)),
    _transformMap(std::move(transformMap)),
    _crosstalk(crosstalk),
    _physicalType(physicalType)
{
    // populate _amplifierMap
    for (auto const & amp : _amplifiers) {
        _amplifierMap.insert(std::make_pair(amp->getName(), amp));
    }
    if (_amplifierMap.size() != _amplifiers.size()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                          "Invalid ampInfoCatalog: not all amplifier names are unique");
    }

    // ensure crosstalk coefficients matrix is square
    if (hasCrosstalk()) {
        auto shape = _crosstalk.getShape();
        assert(shape.size() == 2);  // we've declared this as a 2D array
        if (shape[0] != shape[1]) {
            std::ostringstream os;
            os << "Non-square crosstalk matrix: " << _crosstalk << " for detector \"" << _name << "\"";
            throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
        }
        if (shape[0] != _amplifiers.size()) {
            std::ostringstream os;
            os << "Wrong size crosstalk matrix: " << _crosstalk << " for detector \"" << _name << "\"";
            throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
        }
    }
}

std::vector<lsst::geom::Point2D> Detector::getCorners(CameraSys const &cameraSys) const {
    std::vector<lsst::geom::Point2D> nativeCorners = lsst::geom::Box2D(_bbox).getCorners();
    auto nativeToCameraSys = _transformMap->getTransform(_nativeSys, cameraSys);
    return nativeToCameraSys->applyForward(nativeCorners);
}

std::vector<lsst::geom::Point2D> Detector::getCorners(CameraSysPrefix const &cameraSysPrefix) const {
    return getCorners(makeCameraSys(cameraSysPrefix));
}

lsst::geom::Point2D Detector::getCenter(CameraSys const &cameraSys) const {
    auto ctrPix = lsst::geom::Box2D(_bbox).getCenter();
    auto transform = getTransform(PIXELS, cameraSys);
    return transform->applyForward(ctrPix);
}

lsst::geom::Point2D Detector::getCenter(CameraSysPrefix const &cameraSysPrefix) const {
    return getCenter(makeCameraSys(cameraSysPrefix));
}

Amplifier const & Detector::operator[](std::string const &name) const { return *_get(name); }

std::shared_ptr<Amplifier const> Detector::_get(int i) const {
    if (i < 0) {
        i = _amplifiers.size() + i;
    };
    return _amplifiers.at(i);
}

std::shared_ptr<Amplifier const> Detector::_get(std::string const &name) const {
    auto ampIter = _amplifierMap.find(name);
    if (ampIter == _amplifierMap.end()) {
        std::ostringstream os;
        os << "Unknown amplifier \"" << name << "\"";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    return ampIter->second;
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


class DetectorFactory : public table::io::PersistableFactory {
public:

    DetectorFactory() : PersistableFactory("Detector") {}

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
            amps.push_back(Amplifier::fromRecord(record));
        }

        auto flattenedMatrix = record.get(keys.crosstalk);
        ndarray::Array<float, 2, 2> crosstalk;
        if (!flattenedMatrix.isEmpty()) {
            crosstalk = ndarray::allocate(amps.size(), amps.size());
            ndarray::flatten<1>(crosstalk) = flattenedMatrix;
        }

        // get values for not-always-present fields if present
        const auto physicalType = keys.physicalType.isValid() ? record.get(keys.physicalType) : "";

        return std::make_shared<Detector>(
            record.get(keys.name),
            record.get(keys.id),
            static_cast<DetectorType>(record.get(keys.type)),
            record.get(keys.serial),
            record.get(keys.bbox),
            amps,
            Orientation(
                record.get(keys.fpPosition),
                record.get(keys.refPoint),
                record.get(keys.yaw),
                record.get(keys.pitch),
                record.get(keys.roll)
            ),
            lsst::geom::Extent2D(record.get(keys.pixelSize)),
            archive.get<TransformMap>(record.get(keys.transformMap)),
            crosstalk,
            physicalType
        );
    }

};

DetectorFactory const registration;

} // anonymous

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
