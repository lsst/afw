// -*- lsst-c++ -*-
#include <typeinfo>

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/AmpInfo.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst {
namespace afw {
namespace table {

//-----------------------------------------------------------------------------------------------------------
//----- AmpInfoFitsWriter ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsWriter for AmpInfo - this just sets the AFW_TYPE key to AMPINFO, which should ensure
// we use AmpInfoFitsReader to read it.

namespace {

class AmpInfoFitsWriter : public io::FitsWriter {
public:
    explicit AmpInfoFitsWriter(Fits *fits, int flags) : io::FitsWriter(fits, flags) {}

protected:
    virtual void _writeTable(std::shared_ptr<BaseTable const> const &table, std::size_t nRows);
};

void AmpInfoFitsWriter::_writeTable(std::shared_ptr<BaseTable const> const &t, std::size_t nRows) {
    std::shared_ptr<AmpInfoTable const> table = std::dynamic_pointer_cast<AmpInfoTable const>(t);
    if (!table) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,
                          "Cannot use a AmpInfoFitsWriter on a non-AmpInfo table.");
    }
    io::FitsWriter::_writeTable(table, nRows);
    _fits->writeKey("AFW_TYPE", "AMPINFO", "Tells lsst::afw to load this as a AmpInfo table.");
}

}  // namespace

//-----------------------------------------------------------------------------------------------------------
//----- AmpInfoFitsReader ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsReader for AmpInfoTable/Record - this gets registered with name AMPINFO, so it should get used
// whenever we read a table with AFW_TYPE set to that value.

namespace {

class AmpInfoFitsReader : public io::FitsReader {
public:
    AmpInfoFitsReader() : io::FitsReader("AMPINFO") {}

    virtual std::shared_ptr<BaseTable> makeTable(io::FitsSchemaInputMapper &mapper,
                                                 std::shared_ptr<daf::base::PropertyList> metadata,
                                                 int ioFlags, bool stripMetadata) const {
        std::shared_ptr<AmpInfoTable> table = AmpInfoTable::make(mapper.finalize());
        table->setMetadata(metadata);
        return table;
    }
};

static AmpInfoFitsReader const ampInfoFitsReader;

}  // namespace

//-----------------------------------------------------------------------------------------------------------
//----- AmpInfoTable/Record member function implementations -----------------------------------------------
//-----------------------------------------------------------------------------------------------------------

AmpInfoRecord::AmpInfoRecord(std::shared_ptr<AmpInfoTable> const &table) : BaseRecord(table) {}

AmpInfoRecord::~AmpInfoRecord() = default;

std::shared_ptr<AmpInfoTable> AmpInfoTable::make(Schema const &schema) {
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterError,
                "Schema for AmpInfo must contain at least the keys defined by makeMinimalSchema().");
    }
    return std::shared_ptr<AmpInfoTable>(new AmpInfoTable(schema));
}

AmpInfoTable::AmpInfoTable(Schema const &schema) : BaseTable(schema) {}

AmpInfoTable::AmpInfoTable(AmpInfoTable const &other) : BaseTable(other) {}
// Delegate to copy-constructor for backwards compatibility
AmpInfoTable::AmpInfoTable(AmpInfoTable &&other) : AmpInfoTable(other) {}

AmpInfoTable::~AmpInfoTable() = default;

AmpInfoTable::MinimalSchema::MinimalSchema() {
    name = schema.addField<std::string>("name", "name of amplifier location in camera",
                                        AmpInfoTable::MAX_NAME_LENGTH);
    bboxMin = PointKey<int>::addFields(schema, "bbox_min",
                                       "bbox of amplifier image data on assembled image, min point", "pixel");
    bboxExtent = PointKey<int>::addFields(schema, "bbox_extent",
                                          "bbox of amplifier image data on assembled image, extent", "pixel");
    gain = schema.addField<double>("gain", "amplifier gain", "electron adu^-1");
    saturation = schema.addField<double>(
            "saturation",
            "level above which pixels are considered saturated; use `nan` if no such level applies", "adu");
    suspectLevel = schema.addField<double>(
            "suspectlevel",
            "level above which pixels are considered suspicious, meaning they may be affected by unknown "
            "systematics; for example if non-linearity corrections above a certain level are unstable "
            "then that would be a useful value for suspectLevel; use `nan` if no such level applies",
            "adu");
    readNoise = schema.addField<double>("readnoise", "amplifier read noise", "electron");
    readoutCorner =
            schema.addField<int>("readoutcorner", "readout corner, in the frame of the assembled image");
    linearityCoeffs =
            schema.addField<Array<double> >("linearity_coeffs", "coefficients for linearity fit up to cubic",
                                            AmpInfoTable::MAX_LINEARITY_COEFFS);
    linearityType = schema.addField<std::string>("linearity_type", "type of linearity model",
                                                 AmpInfoTable::MAX_LINEARITY_TYPE_LENGTH);

    // Raw data fields
    hasRawInfo = schema.addField<Flag>(
            "hasrawinfo", "is raw amplifier information available (e.g. untrimmed bounding boxes)?");
    rawBBoxMin = PointKey<int>::addFields(schema, "raw_bbox_min",
                                          "entire amplifier bbox on raw image, min point", "pixel");
    rawBBoxExtent = PointKey<int>::addFields(schema, "raw_bbox_extent",
                                             "entire amplifier bbox on raw image, extent", "pixel");
    rawDataBBoxMin = PointKey<int>::addFields(schema, "raw_databbox_min",
                                              "image data bbox on raw image, min point", "pixel");
    rawDataBBoxExtent = PointKey<int>::addFields(schema, "raw_databbox_extent",
                                                 "image data bbox on raw image, extent", "pixel");
    rawFlipX = schema.addField<Flag>("raw_flip_x", "flip row order to make assembled image?");
    rawFlipY = schema.addField<Flag>("raw_flip_y", "flip column order to make an assembled image?");
    rawXYOffset = PointKey<int>::addFields(
            schema, "raw_xyoffset",
            "offset for assembling a raw CCD image: desired xy0 - raw xy0; 0,0 if raw data comes assembled",
            "pixel");
    rawHorizontalOverscanBBoxMin =
            PointKey<int>::addFields(schema, "raw_horizontaloverscanbbox_min",
                                     "usable horizontal overscan bbox on raw image, min point", "pixel");
    rawHorizontalOverscanBBoxExtent =
            PointKey<int>::addFields(schema, "raw_horizontaloverscanbbox_extent",
                                     "usable horizontal overscan bbox on raw image, extent", "pixel");
    rawVerticalOverscanBBoxMin =
            PointKey<int>::addFields(schema, "raw_verticaloverscanbbox_min",
                                     "usable vertical overscan region raw image, min point", "pixel");
    rawVerticalOverscanBBoxExtent =
            PointKey<int>::addFields(schema, "raw_verticaloverscanbbox_extent",
                                     "usable vertical overscan region raw image, extent", "pixel");
    rawPrescanBBoxMin =
            PointKey<int>::addFields(schema, "raw_prescanbbox_min",
                                     "usable (horizontal) prescan bbox on raw image, min point", "pixel");
    rawPrescanBBoxExtent =
            PointKey<int>::addFields(schema, "raw_prescanbbox_extent",
                                     "usable (horizontal) prescan bbox on raw image, extent", "pixel");
    schema.getCitizen().markPersistent();
}

AmpInfoTable::MinimalSchema &AmpInfoTable::getMinimalSchema() {
    static MinimalSchema it;
    return it;
}

std::shared_ptr<io::FitsWriter> AmpInfoTable::makeFitsWriter(fits::Fits *fitsfile, int flags) const {
    return std::make_shared<AmpInfoFitsWriter>(fitsfile, flags);
}

std::shared_ptr<BaseTable> AmpInfoTable::_clone() const {
    return std::shared_ptr<AmpInfoTable>(new AmpInfoTable(*this));
}

std::shared_ptr<BaseRecord> AmpInfoTable::_makeRecord() {
    return std::shared_ptr<AmpInfoRecord>(new AmpInfoRecord(getSelf<AmpInfoTable>()));
}

//-------------------------------------------------------------------------------------------------
// Getters and Setters
//-------------------------------------------------------------------------------------------------
std::string AmpInfoRecord::getName() const { return get(AmpInfoTable::getNameKey()); }
void AmpInfoRecord::setName(std::string const &name) { set(AmpInfoTable::getNameKey(), name); }

lsst::geom::Box2I AmpInfoRecord::getBBox() const {
    return lsst::geom::Box2I(get(AmpInfoTable::getBBoxMinKey()),
                             lsst::geom::Extent2I(get(AmpInfoTable::getBBoxExtentKey())), false);
}
void AmpInfoRecord::setBBox(lsst::geom::Box2I const &bbox) {
    set(AmpInfoTable::getBBoxMinKey(), bbox.getMin());
    set(AmpInfoTable::getBBoxExtentKey(), lsst::geom::Point2I(bbox.getDimensions()));
}

double AmpInfoRecord::getGain() const { return get(AmpInfoTable::getGainKey()); }
void AmpInfoRecord::setGain(double gain) { set(AmpInfoTable::getGainKey(), gain); }

double AmpInfoRecord::getSaturation() const { return get(AmpInfoTable::getSaturationKey()); }
void AmpInfoRecord::setSaturation(double saturation) { set(AmpInfoTable::getSaturationKey(), saturation); }

double AmpInfoRecord::getSuspectLevel() const { return get(AmpInfoTable::getSuspectLevelKey()); }
void AmpInfoRecord::setSuspectLevel(double suspectLevel) {
    set(AmpInfoTable::getSuspectLevelKey(), suspectLevel);
}

double AmpInfoRecord::getReadNoise() const { return get(AmpInfoTable::getReadNoiseKey()); }
void AmpInfoRecord::setReadNoise(double readNoise) { set(AmpInfoTable::getReadNoiseKey(), readNoise); }

ReadoutCorner AmpInfoRecord::getReadoutCorner() const {
    return static_cast<ReadoutCorner>(get(AmpInfoTable::getReadoutCornerKey()));
}
void AmpInfoRecord::setReadoutCorner(ReadoutCorner readoutCorner) {
    set(AmpInfoTable::getReadoutCornerKey(), readoutCorner);
}

std::vector<double> AmpInfoRecord::getLinearityCoeffs() const {
    Key<Array<double> > coeffKey = AmpInfoTable::getLinearityCoeffsKey();
    return coeffKey.extractVector(*this);
}
void AmpInfoRecord::setLinearityCoeffs(std::vector<double> const &linearityCoeffs) {
    Key<Array<double> > coeffKey = AmpInfoTable::getLinearityCoeffsKey();
    coeffKey.assignVector(*this, linearityCoeffs);
}

std::string AmpInfoRecord::getLinearityType() const { return get(AmpInfoTable::getLinearityTypeKey()); }
void AmpInfoRecord::setLinearityType(std::string const &linearityType) {
    set(AmpInfoTable::getLinearityTypeKey(), linearityType);
}

bool AmpInfoRecord::getHasRawInfo() const { return get(AmpInfoTable::getHasRawInfoKey()); }
void AmpInfoRecord::setHasRawInfo(bool hasrawamplifier) {
    set(AmpInfoTable::getHasRawInfoKey(), hasrawamplifier);
}

lsst::geom::Box2I AmpInfoRecord::getRawBBox() const {
    return lsst::geom::Box2I(get(AmpInfoTable::getRawBBoxMinKey()),
                             lsst::geom::Extent2I(get(AmpInfoTable::getRawBBoxExtentKey())), false);
}
void AmpInfoRecord::setRawBBox(lsst::geom::Box2I const &bbox) {
    set(AmpInfoTable::getRawBBoxMinKey(), bbox.getMin());
    set(AmpInfoTable::getRawBBoxExtentKey(), lsst::geom::Point2I(bbox.getDimensions()));
}

lsst::geom::Box2I AmpInfoRecord::getRawDataBBox() const {
    return lsst::geom::Box2I(get(AmpInfoTable::getRawDataBBoxMinKey()),
                             lsst::geom::Extent2I(get(AmpInfoTable::getRawDataBBoxExtentKey())), false);
}
void AmpInfoRecord::setRawDataBBox(lsst::geom::Box2I const &bbox) {
    set(AmpInfoTable::getRawDataBBoxMinKey(), bbox.getMin());
    set(AmpInfoTable::getRawDataBBoxExtentKey(), lsst::geom::Point2I(bbox.getDimensions()));
}

bool AmpInfoRecord::getRawFlipX() const { return get(AmpInfoTable::getRawFlipXKey()); }
void AmpInfoRecord::setRawFlipX(bool rawFlipX) { set(AmpInfoTable::getRawFlipXKey(), rawFlipX); }

bool AmpInfoRecord::getRawFlipY() const { return get(AmpInfoTable::getRawFlipYKey()); }
void AmpInfoRecord::setRawFlipY(bool rawFlipY) { set(AmpInfoTable::getRawFlipYKey(), rawFlipY); }

lsst::geom::Extent2I AmpInfoRecord::getRawXYOffset() const {
    return lsst::geom::Extent2I(get(AmpInfoTable::getRawXYOffsetKey()));
}
void AmpInfoRecord::setRawXYOffset(lsst::geom::Extent2I const &rawxyoffset) {
    set(AmpInfoTable::getRawXYOffsetKey(), lsst::geom::Point2I(rawxyoffset.asPair()));
}

lsst::geom::Box2I AmpInfoRecord::getRawHorizontalOverscanBBox() const {
    return lsst::geom::Box2I(
            get(AmpInfoTable::getRawHorizontalOverscanBBoxMinKey()),
            lsst::geom::Extent2I(get(AmpInfoTable::getRawHorizontalOverscanBBoxExtentKey())), false);
}
void AmpInfoRecord::setRawHorizontalOverscanBBox(lsst::geom::Box2I const &bbox) {
    set(AmpInfoTable::getRawHorizontalOverscanBBoxMinKey(), bbox.getMin());
    set(AmpInfoTable::getRawHorizontalOverscanBBoxExtentKey(), lsst::geom::Point2I(bbox.getDimensions()));
}

lsst::geom::Box2I AmpInfoRecord::getRawVerticalOverscanBBox() const {
    return lsst::geom::Box2I(get(AmpInfoTable::getRawVerticalOverscanBBoxMinKey()),
                             lsst::geom::Extent2I(get(AmpInfoTable::getRawVerticalOverscanBBoxExtentKey())), false);
}
void AmpInfoRecord::setRawVerticalOverscanBBox(lsst::geom::Box2I const &bbox) {
    set(AmpInfoTable::getRawVerticalOverscanBBoxMinKey(), bbox.getMin());
    set(AmpInfoTable::getRawVerticalOverscanBBoxExtentKey(), lsst::geom::Point2I(bbox.getDimensions()));
}

lsst::geom::Box2I AmpInfoRecord::getRawPrescanBBox() const {
    return lsst::geom::Box2I(get(AmpInfoTable::getRawPrescanBBoxMinKey()),
                             lsst::geom::Extent2I(get(AmpInfoTable::getRawPrescanBBoxExtentKey())), false);
}
void AmpInfoRecord::setRawPrescanBBox(lsst::geom::Box2I const &bbox) {
    set(AmpInfoTable::getRawPrescanBBoxMinKey(), bbox.getMin());
    set(AmpInfoTable::getRawPrescanBBoxExtentKey(), lsst::geom::Point2I(bbox.getDimensions()));
}

template class CatalogT<AmpInfoRecord>;
template class CatalogT<AmpInfoRecord const>;
}  // namespace table
}  // namespace afw
}  // namespace lsst
