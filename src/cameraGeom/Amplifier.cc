// -*- lsst-c++ -*-
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
#include "lsst/afw/cameraGeom/Amplifier.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/aggregates.h"

namespace lsst {
namespace afw {
namespace cameraGeom {

namespace {

/*
 * Private (minimalist) FunctorKey for Box2I that uses _min and _extent, since
 * that's what AmpInfoRecord used in its schema (which predates the new
 * _min+_max BoxKey defined in afw/table/aggregates.h).
 */
class AmpInfoBoxKey : public table::FunctorKey<lsst::geom::Box2I> {
public:

    /**
     *  Add _min_x, _min_y, _extent_x, _extent_y fields to a Schema, and
     *  return an AmpInfoBoxKey that points to them.
     *
     *  @param[in,out] schema  Schema to add fields to.
     *  @param[in]     name    Name prefix for all fields; suffixes above will
     *                         be appended to this to form the full field
     *                         names.  For example, if `name == "b"`, the
     *                         fields added will be "b_min_x", "b_min_y",
     *                         "b_max_x", and "b_max_y".
     *  @param[in]     doc     String used as the documentation for the fields.
     *  @param[in]     unit    String used as the unit for all fields.
     *
     */
    static AmpInfoBoxKey addFields(table::Schema& schema, std::string const& name, std::string const& doc,
                                   std::string const& unit) {
        AmpInfoBoxKey result;
        result._min = table::Point2IKey::addFields(
            schema,
            schema.join(name, "min"),
            doc + ", min point",
            unit
        );
        result._dimensions = table::Point2IKey::addFields(
            schema,
            schema.join(name, "extent"),
            doc + ", extent",
            unit
        );
        return result;
    }

    // Default constructor; instance will not be usable unless subsequently assigned to.
    AmpInfoBoxKey() noexcept = default;

    /*
     *  Construct from a subschema, assuming _min_x, _max_x, _min_y, _max_y subfields
     *
     *  If a schema has "a_min_x" and "a_min_x" (etc) fields, this constructor allows you to construct
     *  a BoxKey via:
     *
     *      BoxKey<Box> k(schema["a"]);
     */
    AmpInfoBoxKey(table::SubSchema const& s) : _min(s["min"]), _dimensions(s["extent"]) {}

    AmpInfoBoxKey(AmpInfoBoxKey const&) noexcept = default;
    AmpInfoBoxKey(AmpInfoBoxKey&&) noexcept = default;
    AmpInfoBoxKey& operator=(AmpInfoBoxKey const&) noexcept = default;
    AmpInfoBoxKey& operator=(AmpInfoBoxKey&&) noexcept = default;
    ~AmpInfoBoxKey() noexcept override = default;

    // Get a Box from the given record
    lsst::geom::Box2I get(table::BaseRecord const& record) const override {
        return lsst::geom::Box2I(record.get(_min),
                                 lsst::geom::Extent2I(record.get(_dimensions)));
    }

    // Set a Box in the given record
    void set(table::BaseRecord& record, lsst::geom::Box2I const& value) const override {
        record.set(_min, value.getMin());
        record.set(_dimensions, lsst::geom::Point2I(value.getDimensions()));
    }

    // Return True if both the min and max PointKeys are valid.
    bool isValid() const noexcept { return _min.isValid() && _dimensions.isValid(); }

private:
    table::Point2IKey _min;
    table::Point2IKey _dimensions;
};


struct RecordSchemaHelper {
    table::Schema schema;
    table::Key<std::string> name;
    AmpInfoBoxKey bbox;
    table::Key<double> gain;
    table::Key<double> saturation;
    table::Key<double> suspectLevel;
    table::Key<double> readNoise;
    table::Key<int> readoutCorner;
    table::Key<table::Array<double> > linearityCoeffs;
    table::Key<std::string> linearityType;
    table::Key<table::Flag> hasRawInfo;
    AmpInfoBoxKey rawBBox;
    AmpInfoBoxKey rawDataBBox;
    table::Key<table::Flag> rawFlipX;
    table::Key<table::Flag> rawFlipY;
    table::PointKey<int> rawXYOffset;
    AmpInfoBoxKey rawHorizontalOverscanBBox;
    AmpInfoBoxKey rawVerticalOverscanBBox;
    AmpInfoBoxKey rawHorizontalPrescanBBox;
    table::Key<double> linearityThreshold;
    table::Key<double> linearityMaximum;
    table::Key<std::string> linearityUnits;

    static RecordSchemaHelper const & getMinimal() {
        static RecordSchemaHelper const instance;
        return instance;
    }

    RecordSchemaHelper(table::Schema const & existing) :
        schema(existing),
        name(schema["name"]),
        bbox(schema["bbox"]),
        gain(schema["gain"]),
        saturation(schema["saturation"]),
        suspectLevel(schema["suspectlevel"]),
        readNoise(schema["readnoise"]),
        readoutCorner(schema["readoutcorner"]),
        linearityCoeffs(schema["linearity_coeffs"]),
        linearityType(schema["linearity_type"]),
        hasRawInfo(schema["hasrawinfo"]),
        rawBBox(schema["raw_bbox"]),
        rawDataBBox(schema["raw_databbox"]),
        rawFlipX(schema["raw_flip_x"]),
        rawFlipY(schema["raw_flip_y"]),
        rawXYOffset(schema["raw_xyoffset"]),
        rawHorizontalOverscanBBox(schema["raw_horizontaloverscanbbox"]),
        rawVerticalOverscanBBox(schema["raw_verticaloverscanbbox"]),
        rawHorizontalPrescanBBox(schema["raw_prescanbbox"])
    {
        auto setKeyIfPresent = [this](auto & key, std::string const & name) {
            try {
                key = schema[name];
            } catch (pex::exceptions::NotFoundError &) {}
        };
        // These fields were not part of the original AmpInfoRecord minimal
        // schema, but were frequently used and are now present on all
        // Amplifier objects, even if unused.
        // Unfortunately they use a different naming convention than the
        // others, but as humans will rarely interact with the record form
        // going forward this is not a big deal.
        setKeyIfPresent(linearityThreshold, "linearityThreshold");
        setKeyIfPresent(linearityMaximum, "linearityMaximum");
        setKeyIfPresent(linearityUnits, "linearityUnits");
    }

private:

    RecordSchemaHelper() :
        schema(),
        name(schema.addField<std::string>("name", "name of amplifier location in camera", "", 0)),
        bbox(AmpInfoBoxKey::addFields(
             schema, "bbox", "bbox of amplifier image data on assembled image", "pixel")),
        gain(schema.addField<double>("gain", "amplifier gain", "electron adu^-1")),
        saturation(schema.addField<double>(
            "saturation",
            "level above which pixels are considered saturated; use `nan` if no such level applies",
            "adu")),
        suspectLevel(schema.addField<double>(
            "suspectlevel",
            "level above which pixels are considered suspicious, meaning they may be affected by unknown "
            "systematics; for example if non-linearity corrections above a certain level are unstable "
            "then that would be a useful value for suspectLevel; use `nan` if no such level applies",
            "adu")),
        readNoise(schema.addField<double>("readnoise", "amplifier read noise", "electron")),
        readoutCorner(
            schema.addField<int>("readoutcorner", "readout corner, in the frame of the assembled image")),
        linearityCoeffs(
            schema.addField<table::Array<double> >("linearity_coeffs",
                                                   "coefficients for linearity fit up to cubic", "", 0)),
        linearityType(schema.addField<std::string>("linearity_type", "type of linearity model", "", 0)),
        hasRawInfo(schema.addField<table::Flag>(
            "hasrawinfo", "is raw amplifier information available (e.g. untrimmed bounding boxes)?")),
        rawBBox(AmpInfoBoxKey::addFields(schema, "raw_bbox",
                                         "entire amplifier bbox on raw image", "pixel")),
        rawDataBBox(AmpInfoBoxKey::addFields(schema, "raw_databbox",
                                             "image data bbox on raw image", "pixel")),
        rawFlipX(schema.addField<table::Flag>("raw_flip_x", "flip row order to make assembled image?")),
        rawFlipY(schema.addField<table::Flag>("raw_flip_y", "flip column order to make an assembled image?")),
        rawXYOffset(table::Point2IKey::addFields(
            schema, "raw_xyoffset",
            "offset for assembling a raw CCD image: desired xy0 - raw xy0; 0,0 if raw data comes assembled",
            "pixel")),
        rawHorizontalOverscanBBox(
            AmpInfoBoxKey::addFields(schema, "raw_horizontaloverscanbbox",
                                     "usable horizontal overscan bbox on raw image", "pixel")),
        rawVerticalOverscanBBox(
            AmpInfoBoxKey::addFields(schema, "raw_verticaloverscanbbox",
                                     "usable vertical overscan region raw image", "pixel")),
        rawHorizontalPrescanBBox(
            AmpInfoBoxKey::addFields(schema, "raw_prescanbbox",
                                     "usable (horizontal) prescan bbox on raw image", "pixel")),
        linearityThreshold(schema.addField<double>("linearityThreshold", "TODO! NEVER DOCUMENTED")),
        linearityMaximum(schema.addField<double>("linearityMaximum", "TODO! NEVER DOCUMENTED")),
        linearityUnits(schema.addField<std::string>("linearityUnits", "TODO! NEVER DOCUMENTED", "", 0))
    {
        schema.getCitizen().markPersistent();
    }

};


class FrozenAmplifier final : public Amplifier {
public:

    explicit FrozenAmplifier(Fields const & fields) : _fields(fields) {}

    FrozenAmplifier(FrozenAmplifier const &) = delete;
    FrozenAmplifier(FrozenAmplifier &&) = delete;

    FrozenAmplifier & operator=(FrozenAmplifier const &) = delete;
    FrozenAmplifier & operator=(FrozenAmplifier &&) = delete;

    ~FrozenAmplifier() noexcept override = default;

protected:

    Fields const & getFields() const override { return _fields; }

private:

    Fields const _fields;
};

} // anonyomous


table::Schema Amplifier::getRecordSchema() {
    return RecordSchemaHelper::getMinimal().schema;
}

Amplifier::Builder Amplifier::rebuild() const { return Builder(*this); }

Amplifier::Builder::Builder(Amplifier const & other) : _fields(other.getFields()) {}

Amplifier::Builder & Amplifier::Builder::operator=(Amplifier const & other) {
    _fields = other.getFields();
    return *this;
}

std::shared_ptr<Amplifier const> Amplifier::Builder::finish() const {
    return std::make_shared<FrozenAmplifier>(_fields);
}

Amplifier::Builder Amplifier::Builder::fromRecord(table::BaseRecord const & record) {
    auto const helper = RecordSchemaHelper(record.getSchema());
    Builder result;
    result.setName(record.get(helper.name));
    result.setBBox(record.get(helper.bbox));
    result.setGain(record.get(helper.gain));
    result.setReadNoise(record.get(helper.readNoise));
    result.setSaturation(record.get(helper.saturation));
    result.setSuspectLevel(record.get(helper.suspectLevel));
    result.setReadoutCorner(static_cast<ReadoutCorner>(record.get(helper.readoutCorner)));
    result.setLinearityCoeffs(ndarray::copy(record.get(helper.linearityCoeffs)));
    result.setLinearityType(record.get(helper.linearityType));
    result.setRawBBox(record.get(helper.rawBBox));
    result.setRawDataBBox(record.get(helper.rawDataBBox));
    result.setRawFlipX(record.get(helper.rawFlipX));
    result.setRawFlipY(record.get(helper.rawFlipY));
    result.setRawXYOffset(lsst::geom::Extent2I(record.get(helper.rawXYOffset)));
    result.setRawHorizontalOverscanBBox(record.get(helper.rawHorizontalOverscanBBox));
    result.setRawVerticalOverscanBBox(record.get(helper.rawVerticalOverscanBBox));
    result.setRawHorizontalPrescanBBox(record.get(helper.rawHorizontalPrescanBBox));
    // Set not-always-present fields only when present.  While it's usually
    // preferable to use the public setter methods (as above), passing member
    // function pointers through the lambda below is sufficiently unpleasant
    // that we just set the private member directly.
    auto setIfValid = [&record](auto & member, auto & key) {
        if (key.isValid()) {
            member = record.get(key);
        }
    };
    setIfValid(result._fields.linearityThreshold, helper.linearityThreshold);
    setIfValid(result._fields.linearityMaximum, helper.linearityMaximum);
    setIfValid(result._fields.linearityUnits, helper.linearityUnits);
    return result;
}

void Amplifier::toRecord(table::BaseRecord & record) const {
    auto const helper = RecordSchemaHelper(record.getSchema());
    auto const & fields = getFields();
    record.set(helper.name, fields.name);
    record.set(helper.bbox, fields.bbox);
    record.set(helper.gain, fields.gain);
    record.set(helper.readNoise, fields.readNoise);
    record.set(helper.saturation, fields.saturation);
    record.set(helper.suspectLevel, fields.suspectLevel);
    record.set(helper.readoutCorner, static_cast<int>(fields.readoutCorner));
    record.set(helper.rawBBox, fields.rawBBox);
    record.set(helper.rawDataBBox, fields.rawDataBBox);
    record.set(helper.rawFlipX, fields.rawFlipX);
    record.set(helper.rawFlipY, fields.rawFlipY);
    record.set(helper.rawXYOffset, lsst::geom::Point2I(fields.rawXYOffset));
    record.set(helper.rawHorizontalOverscanBBox, fields.rawHorizontalOverscanBBox);
    record.set(helper.rawVerticalOverscanBBox, fields.rawVerticalOverscanBBox);
    record.set(helper.rawHorizontalPrescanBBox, fields.rawHorizontalPrescanBBox);
    // Set not-always-present fields only when present.
    auto setIfValid = [this, &record](auto value, auto & key) {
        if (key.isValid()) {
            record.set(key, value);
        }
    };
    setIfValid(fields.linearityThreshold, helper.linearityThreshold);
    setIfValid(fields.linearityMaximum, helper.linearityMaximum);
    setIfValid(fields.linearityUnits, helper.linearityUnits);
}

}  // namespace table
}  // namespace afw
}  // namespace lsst
