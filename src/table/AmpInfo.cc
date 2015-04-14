// -*- lsst-c++ -*-
#include <typeinfo>

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/AmpInfo.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

//-----------------------------------------------------------------------------------------------------------
//----- Private AmpInfoTable/Record classes ---------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// These private derived classes are what you actually get when you do AmpInfoTable::make; like the
// private classes in BaseTable.cc, it's more convenient to have an extra set of trivial derived
// classes than to do a lot of friending.

namespace {

class AmpInfoTableImpl;

class AmpInfoRecordImpl : public AmpInfoRecord {
public:

    explicit AmpInfoRecordImpl(PTR(AmpInfoTable) const & table) : AmpInfoRecord(table) {}

};

class AmpInfoTableImpl : public AmpInfoTable {
public:

    explicit AmpInfoTableImpl(Schema const & schema) :
        AmpInfoTable(schema)
    {}

    AmpInfoTableImpl(AmpInfoTableImpl const & other) : AmpInfoTable(other) {}

private:

    virtual PTR(BaseTable) _clone() const {
        return boost::make_shared<AmpInfoTableImpl>(*this);
    }

    virtual PTR(BaseRecord) _makeRecord() {
        return boost::make_shared<AmpInfoRecordImpl>(getSelf<AmpInfoTableImpl>());
    }

};

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- AmpInfoFitsWriter ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsWriter for AmpInfo - this just sets the AFW_TYPE key to AMPINFO, which should ensure
// we use AmpInfoFitsReader to read it.

namespace {

class AmpInfoFitsWriter : public io::FitsWriter {
public:

    explicit AmpInfoFitsWriter(Fits * fits, int flags) : io::FitsWriter(fits, flags) {}

protected:

    virtual void _writeTable(CONST_PTR(BaseTable) const & table, std::size_t nRows);

};

void AmpInfoFitsWriter::_writeTable(CONST_PTR(BaseTable) const & t, std::size_t nRows) {
    CONST_PTR(AmpInfoTable) table = boost::dynamic_pointer_cast<AmpInfoTable const>(t);
    if (!table) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicError,
            "Cannot use a AmpInfoFitsWriter on a non-AmpInfo table."
        );
    }
    io::FitsWriter::_writeTable(table, nRows);
    _fits->writeKey("AFW_TYPE", "AMPINFO", "Tells lsst::afw to load this as a AmpInfo table.");
}

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- AmpInfoFitsReader ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsReader for AmpInfoTable/Record - this gets registered with name AMPINFO, so it should get used
// whenever we read a table with AFW_TYPE set to that value.

namespace {

class AmpInfoFitsReader : public io::FitsReader {
public:

    AmpInfoFitsReader() : io::FitsReader("AMPINFO") {}

    virtual PTR(BaseTable) makeTable(
        io::FitsSchemaInputMapper & mapper,
        PTR(daf::base::PropertyList) metadata,
        int ioFlags,
        bool stripMetadata
    ) const {
        PTR(AmpInfoTable) table = AmpInfoTable::make(mapper.finalize());
        table->setMetadata(metadata);
        return table;
    }

};

static AmpInfoFitsReader const ampInfoFitsReader;

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- AmpInfoTable/Record member function implementations -----------------------------------------------
//-----------------------------------------------------------------------------------------------------------

AmpInfoRecord::AmpInfoRecord(PTR(AmpInfoTable) const & table) : BaseRecord(table) {}

PTR(AmpInfoTable) AmpInfoTable::make(Schema const & schema) {
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError,
            "Schema for AmpInfo must contain at least the keys defined by makeMinimalSchema()."
        );
    }
    return boost::make_shared<AmpInfoTableImpl>(schema);
}

AmpInfoTable::AmpInfoTable(Schema const & schema) :
    BaseTable(schema) {}

AmpInfoTable::AmpInfoTable(AmpInfoTable const & other) :
    BaseTable(other) {}

AmpInfoTable::MinimalSchema::MinimalSchema() {
    name = schema.addField<std::string>(
        "name",
        "name of amplifier location in camera",
        AmpInfoTable::MAX_NAME_LENGTH);
    bboxMin = schema.addField< Point<int> >(
        "bbox.min",
        "bbox of amplifier image data on assembled image, min point",
        "pixels");
    bboxExtent = schema.addField< Point<int> >(
        "bbox.extent",
        "bbox of amplifier image data on assembled image, extent",
        "pixels");
    gain = schema.addField<double>(
        "gain",
        "amplifier gain in e-/ADU",
        "e-/ADU");
    saturation = schema.addField<int>(
        "saturation",
        "saturation value, in ADU",
        "ADU");
    readNoise = schema.addField<double>(
        "readnoise",
        "amplifier read noise, in e-",
        "e-");
    readoutCorner = schema.addField<int>(
        "readoutcorner",
        "readout corner, in the frame of the assembled image");
    linearityCoeffs = schema.addField< Array<double> >(
        "linearity.coeffs",
        "coefficients for linearity fit up to cubic",
        AmpInfoTable::MAX_LINEARITY_COEFFS);
    linearityType = schema.addField<std::string>(
        "linearity.type",
        "type of linearity model",
        AmpInfoTable::MAX_LINEARITY_TYPE_LENGTH);

    // Raw data fields
    hasRawInfo = schema.addField<Flag>(
        "hasrawinfo",
        "is raw amplifier information available (e.g. untrimmed bounding boxes)?");
    rawBBoxMin = schema.addField< Point<int> >(
        "raw.bbox.min",
        "entire amplifier bbox on raw image, min point",
        "pixels");
    rawBBoxExtent = schema.addField< Point<int> >(
        "raw.bbox.extent",
        "entire amplifier bbox on raw image, extent",
        "pixels");
    rawDataBBoxMin = schema.addField< Point<int> >(
        "raw.databbox.min",
        "image data bbox on raw image, min point",
        "pixels");
    rawDataBBoxExtent = schema.addField< Point<int> >(
        "raw.databbox.extent",
        "image data bbox on raw image, extent",
        "pixels");
    rawFlipX = schema.addField<Flag>(
        "raw.flip.x",
        "flip row order to make assembled image?");
    rawFlipY = schema.addField<Flag>(
        "raw.flip.y",
        "flip column order to make an assembled image?");
    rawXYOffset = schema.addField< Point<int> >(
        "raw.xyoffset",
        "offset for assembling a raw CCD image: desired xy0 - raw xy0; 0,0 if raw data comes assembled",
        "pixels");
    rawHorizontalOverscanBBoxMin = schema.addField< Point<int> >(
        "raw.horizontaloverscanbbox.min",
        "usable horizontal overscan bbox on raw image, min point",
        "pixels");
    rawHorizontalOverscanBBoxExtent = schema.addField< Point<int> >(
        "raw.horizontaloverscanbbox.extent",
        "usable horizontal overscan bbox on raw image, extent",
        "pixels");
    rawVerticalOverscanBBoxMin = schema.addField< Point<int> >(
        "raw.verticaloverscanbbox.min",
        "usable vertical overscan region raw image, min point",
        "pixels");
    rawVerticalOverscanBBoxExtent = schema.addField< Point<int> >(
        "raw.verticaloverscanbbox.extent",
        "usable vertical overscan region raw image, extent",
        "pixels");
    rawPrescanBBoxMin = schema.addField< Point<int> >(
        "raw.prescanbbox.min",
        "usable (horizontal) prescan bbox on raw image, min point",
        "pixels");
    rawPrescanBBoxExtent = schema.addField< Point<int> >(
        "raw.prescanbbox.extent",
        "usable (horizontal) prescan bbox on raw image, extent",
        "pixels");
    schema.getCitizen().markPersistent();
}

AmpInfoTable::MinimalSchema & AmpInfoTable::getMinimalSchema() {
    static MinimalSchema it;
    return it;
}

PTR(io::FitsWriter)
AmpInfoTable::makeFitsWriter(fits::Fits * fitsfile, int flags) const {
    return boost::make_shared<AmpInfoFitsWriter>(fitsfile, flags);
}
//-------------------------------------------------------------------------------------------------
// Getters and Setters
//-------------------------------------------------------------------------------------------------
std::string AmpInfoRecord::getName() const { return get(AmpInfoTable::getNameKey()); }
void AmpInfoRecord::setName(std::string const &name) { set(AmpInfoTable::getNameKey(), name); }

geom::Box2I AmpInfoRecord::getBBox() const {
    return geom::Box2I(
        get(AmpInfoTable::getBBoxMinKey()),
        geom::Extent2I(get(AmpInfoTable::getBBoxExtentKey()))
    );
}
void AmpInfoRecord::setBBox(geom::Box2I const &bbox) {
    set(AmpInfoTable::getBBoxMinKey(), bbox.getMin());
    set(AmpInfoTable::getBBoxExtentKey(), geom::Point2I(bbox.getDimensions()));
}

double AmpInfoRecord::getGain() const { return get(AmpInfoTable::getGainKey()); }
void AmpInfoRecord::setGain(double gain) { set(AmpInfoTable::getGainKey(), gain); }

int AmpInfoRecord::getSaturation() const { return get(AmpInfoTable::getSaturationKey()); }
void AmpInfoRecord::setSaturation(int saturation) { set(AmpInfoTable::getSaturationKey(), saturation); }

double AmpInfoRecord::getReadNoise() const { return get(AmpInfoTable::getReadNoiseKey()); }
void AmpInfoRecord::setReadNoise(double readNoise) { set(AmpInfoTable::getReadNoiseKey(), readNoise); }

ReadoutCorner AmpInfoRecord::getReadoutCorner() const {
     return static_cast<ReadoutCorner>(get(AmpInfoTable::getReadoutCornerKey()));
 }
void AmpInfoRecord::setReadoutCorner(ReadoutCorner readoutCorner) {
    set(AmpInfoTable::getReadoutCornerKey(), readoutCorner);
}

std::vector<double> AmpInfoRecord::getLinearityCoeffs() const {
    Key< Array<double> > coeffKey = AmpInfoTable::getLinearityCoeffsKey();
    return coeffKey.extractVector(*this);
}
void AmpInfoRecord::setLinearityCoeffs(std::vector<double> const &linearityCoeffs) {
    Key< Array<double> > coeffKey = AmpInfoTable::getLinearityCoeffsKey();
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

geom::Box2I AmpInfoRecord::getRawBBox() const {
    return geom::Box2I(
        get(AmpInfoTable::getRawBBoxMinKey()),
        geom::Extent2I(get(AmpInfoTable::getRawBBoxExtentKey()))
    );
}
void AmpInfoRecord::setRawBBox(geom::Box2I const &bbox) {
    set(AmpInfoTable::getRawBBoxMinKey(), bbox.getMin());
    set(AmpInfoTable::getRawBBoxExtentKey(), geom::Point2I(bbox.getDimensions()));
}

geom::Box2I AmpInfoRecord::getRawDataBBox() const {
    return geom::Box2I(
        get(AmpInfoTable::getRawDataBBoxMinKey()),
        geom::Extent2I(get(AmpInfoTable::getRawDataBBoxExtentKey()))
    );
}
void AmpInfoRecord::setRawDataBBox(geom::Box2I const &bbox) {
    set(AmpInfoTable::getRawDataBBoxMinKey(), bbox.getMin());
    set(AmpInfoTable::getRawDataBBoxExtentKey(), geom::Point2I(bbox.getDimensions()));
}

bool AmpInfoRecord::getRawFlipX() const { return get(AmpInfoTable::getRawFlipXKey()); }
void AmpInfoRecord::setRawFlipX(bool rawFlipX) { set(AmpInfoTable::getRawFlipXKey(), rawFlipX); }

bool AmpInfoRecord::getRawFlipY() const { return get(AmpInfoTable::getRawFlipYKey()); }
void AmpInfoRecord::setRawFlipY(bool rawFlipY) { set(AmpInfoTable::getRawFlipYKey(), rawFlipY); }

geom::Extent2I AmpInfoRecord::getRawXYOffset() const {
    return geom::Extent2I(get(AmpInfoTable::getRawXYOffsetKey()));
}
void AmpInfoRecord::setRawXYOffset(geom::Extent2I const &rawxyoffset) {
    set(AmpInfoTable::getRawXYOffsetKey(), geom::Point2I(rawxyoffset.asPair()));
}

geom::Box2I AmpInfoRecord::getRawHorizontalOverscanBBox() const {
    return geom::Box2I(
        get(AmpInfoTable::getRawHorizontalOverscanBBoxMinKey()),
        geom::Extent2I(get(AmpInfoTable::getRawHorizontalOverscanBBoxExtentKey()))
    );
}
void AmpInfoRecord::setRawHorizontalOverscanBBox(geom::Box2I const &bbox) {
    set(AmpInfoTable::getRawHorizontalOverscanBBoxMinKey(), bbox.getMin());
    set(AmpInfoTable::getRawHorizontalOverscanBBoxExtentKey(), geom::Point2I(bbox.getDimensions()));
}

geom::Box2I AmpInfoRecord::getRawVerticalOverscanBBox() const {
    return geom::Box2I(
        get(AmpInfoTable::getRawVerticalOverscanBBoxMinKey()),
        geom::Extent2I(get(AmpInfoTable::getRawVerticalOverscanBBoxExtentKey()))
    );
}
void AmpInfoRecord::setRawVerticalOverscanBBox(geom::Box2I const &bbox) {
    set(AmpInfoTable::getRawVerticalOverscanBBoxMinKey(), bbox.getMin());
    set(AmpInfoTable::getRawVerticalOverscanBBoxExtentKey(), geom::Point2I(bbox.getDimensions()));
}

geom::Box2I AmpInfoRecord::getRawPrescanBBox() const {
    return geom::Box2I(
        get(AmpInfoTable::getRawPrescanBBoxMinKey()),
        geom::Extent2I(get(AmpInfoTable::getRawPrescanBBoxExtentKey()))
    );
}
void AmpInfoRecord::setRawPrescanBBox(geom::Box2I const &bbox) {
    set(AmpInfoTable::getRawPrescanBBoxMinKey(), bbox.getMin());
    set(AmpInfoTable::getRawPrescanBBoxExtentKey(), geom::Point2I(bbox.getDimensions()));
}

template class CatalogT<AmpInfoRecord>;
template class CatalogT<AmpInfoRecord const>;

}}} // namespace lsst::afw::table
