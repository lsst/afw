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
            lsst::pex::exceptions::LogicErrorException,
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

    explicit AmpInfoFitsReader(Fits * fits, PTR(io::InputArchive) archive, int flags) :
        io::FitsReader(fits, archive, flags) {}

protected:

    virtual PTR(BaseTable) _readTable();

};

PTR(BaseTable) AmpInfoFitsReader::_readTable() {
    PTR(daf::base::PropertyList) metadata = boost::make_shared<daf::base::PropertyList>();
    _fits->readMetadata(*metadata, true);
    Schema schema(*metadata, true);
    PTR(AmpInfoTable) table = AmpInfoTable::make(schema);
    _startRecords(*table);
    if (metadata->exists("AFW_TYPE")) metadata->remove("AFW_TYPE");
    table->setMetadata(metadata);
    return table;
}

// registers the reader so FitsReader::make can use it.
static io::FitsReader::FactoryT<AmpInfoFitsReader> referenceFitsReaderFactory("AMPINFO");

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- AmpInfoTable/Record member function implementations -----------------------------------------------
//-----------------------------------------------------------------------------------------------------------

AmpInfoRecord::AmpInfoRecord(PTR(AmpInfoTable) const & table) : BaseRecord(table) {}

PTR(AmpInfoTable) AmpInfoTable::make(Schema const & schema) {
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
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
    bboxMax = schema.addField< Point<int> >(
        "bbox.max",
        "bbox of amplifier image data on assembled image, max point",
        "pixels");
    gain = schema.addField<double>(
        "gain",
        "amplifier gain in e-/ADU",
        "e-/ADU");
    readNoise = schema.addField<double>(
        "readnoise",
        "amplifier read noise, in e-",
        "e-");
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
    rawBBoxMax = schema.addField< Point<int> >(
        "raw.bbox.max",
        "entire amplifier bbox on raw image, max point",
        "pixels");
    rawDataBBoxMin = schema.addField< Point<int> >(
        "raw.databbox.min",
        "image data bbox on raw image, min point",
        "pixels");
    rawDataBBoxMax = schema.addField< Point<int> >(
        "raw.databbox.max",
        "image data bbox on raw image, max point",
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
    rawHorizontalOverscanBBoxMax = schema.addField< Point<int> >(
        "raw.horizontaloverscanbbox.max", 
        "usable horizontal overscan bbox on raw image, max point",
        "pixels");
    rawVerticalOverscanBBoxMin = schema.addField< Point<int> >(
        "raw.verticaloverscanbbox.min", 
        "usable vertical overscan region raw image, min point",
        "pixels");
    rawVerticalOverscanBBoxMax = schema.addField< Point<int> >(
        "raw.verticaloverscanbbox.max", 
        "usable vertical overscan region raw image, max point",
        "pixels");
    rawPrescanBBoxMin = schema.addField< Point<int> >(
        "raw.prescanbbox.min",
        "usable (horizontal) prescan bbox on raw image, min point",
        "pixels");
    rawPrescanBBoxMax = schema.addField< Point<int> >(
        "raw.prescanbbox.max",
        "usable (horizontal) prescan bbox on raw image, max point",
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
    return geom::Box2I(get(AmpInfoTable::getBBoxMinKey()), get(AmpInfoTable::getBBoxMaxKey())); 
}
void AmpInfoRecord::setBBox(geom::Box2I const &bbox) { 
    set(AmpInfoTable::getBBoxMinKey(), bbox.getMin()); 
    set(AmpInfoTable::getBBoxMaxKey(), bbox.getMax()); 
}

double AmpInfoRecord::getGain() const { return get(AmpInfoTable::getGainKey()); }
void AmpInfoRecord::setGain(double gain) { set(AmpInfoTable::getGainKey(), gain); }

double AmpInfoRecord::getReadNoise() const { return get(AmpInfoTable::getReadNoiseKey()); }
void AmpInfoRecord::setReadNoise(double readNoise) { set(AmpInfoTable::getReadNoiseKey(), readNoise); }

std::vector<double> AmpInfoRecord::getLinearityCoeffs() const { 
    Key< Array<double> > coeffKey = AmpInfoTable::getLinearityCoeffsKey();
    return coeffKey.extractVector(*this);
}
void AmpInfoRecord::setLinearityCoeffs(std::vector<double> const &linearitycoeffs) { 
    Key< Array<double> > coeffKey = AmpInfoTable::getLinearityCoeffsKey();
    coeffKey.assignVector(*this, linearitycoeffs);
}

std::string AmpInfoRecord::getLinearityType() const { return get(AmpInfoTable::getLinearityTypeKey()); }
void AmpInfoRecord::setLinearityType(std::string const &linearitytype) { set(AmpInfoTable::getLinearityTypeKey(), linearitytype); }

bool AmpInfoRecord::getHasRawInfo() const { return get(AmpInfoTable::getHasRawInfoKey()); }
void AmpInfoRecord::setHasRawInfo(bool hasrawamplifier) { set(AmpInfoTable::getHasRawInfoKey(), hasrawamplifier); }

geom::Box2I AmpInfoRecord::getRawBBox() const { 
    return geom::Box2I(get(AmpInfoTable::getRawBBoxMinKey()), get(AmpInfoTable::getRawBBoxMaxKey())); 
}
void AmpInfoRecord::setRawBBox(geom::Box2I const &bbox) { 
    set(AmpInfoTable::getRawBBoxMinKey(), bbox.getMin()); 
    set(AmpInfoTable::getRawBBoxMaxKey(), bbox.getMax()); 
}

geom::Box2I AmpInfoRecord::getRawDataBBox() const { 
    return geom::Box2I(get(AmpInfoTable::getRawDataBBoxMinKey()), get(AmpInfoTable::getRawDataBBoxMaxKey())); 
}
void AmpInfoRecord::setRawDataBBox(geom::Box2I const &bbox) { 
    set(AmpInfoTable::getRawDataBBoxMinKey(), bbox.getMin()); 
    set(AmpInfoTable::getRawDataBBoxMaxKey(), bbox.getMax()); 
}

bool AmpInfoRecord::getRawFlipX() const { return get(AmpInfoTable::getRawFlipXKey()); }
void AmpInfoRecord::setRawFlipX(bool rawFlipX) { set(AmpInfoTable::getRawFlipXKey(), rawFlipX); }

bool AmpInfoRecord::getRawFlipY() const { return get(AmpInfoTable::getRawFlipYKey()); }
void AmpInfoRecord::setRawFlipY(bool rawFlipY) { set(AmpInfoTable::getRawFlipYKey(), rawFlipY); }

geom::Extent2I AmpInfoRecord::getRawXYOffset() const { return geom::Extent2I(get(AmpInfoTable::getRawXYOffsetKey())); }
void AmpInfoRecord::setRawXYOffset(geom::Extent2I const &rawxyoffset) { 
    set(AmpInfoTable::getRawXYOffsetKey(), geom::Point2I(rawxyoffset.asPair())); 
}

geom::Box2I AmpInfoRecord::getRawHorizontalOverscanBBox() const { 
    return geom::Box2I(get(AmpInfoTable::getRawHorizontalOverscanBBoxMinKey()), get(AmpInfoTable::getRawHorizontalOverscanBBoxMaxKey())); 
}
void AmpInfoRecord::setRawHorizontalOverscanBBox(geom::Box2I const &bbox) { 
    set(AmpInfoTable::getRawHorizontalOverscanBBoxMinKey(), bbox.getMin()); 
    set(AmpInfoTable::getRawHorizontalOverscanBBoxMaxKey(), bbox.getMax()); 
}

geom::Box2I AmpInfoRecord::getRawVerticalOverscanBBox() const { 
    return geom::Box2I(get(AmpInfoTable::getRawVerticalOverscanBBoxMinKey()), get(AmpInfoTable::getRawVerticalOverscanBBoxMaxKey())); 
}
void AmpInfoRecord::setRawVerticalOverscanBBox(geom::Box2I const &bbox) { 
    set(AmpInfoTable::getRawVerticalOverscanBBoxMinKey(), bbox.getMin()); 
    set(AmpInfoTable::getRawVerticalOverscanBBoxMaxKey(), bbox.getMax()); 
}

geom::Box2I AmpInfoRecord::getRawPrescanBBox() const { 
    return geom::Box2I(get(AmpInfoTable::getRawPrescanBBoxMinKey()), get(AmpInfoTable::getRawPrescanBBoxMaxKey())); 
}
void AmpInfoRecord::setRawPrescanBBox(geom::Box2I const &bbox) { 
    set(AmpInfoTable::getRawPrescanBBoxMinKey(), bbox.getMin()); 
    set(AmpInfoTable::getRawPrescanBBoxMaxKey(), bbox.getMax()); 
}

template class CatalogT<AmpInfoRecord>;
template class CatalogT<AmpInfoRecord const>;

}}} // namespace lsst::afw::table
