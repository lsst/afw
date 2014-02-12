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

    explicit AmpInfoTableImpl(Schema const & schema, PTR(IdFactory) const & idFactory) : 
        AmpInfoTable(schema, idFactory)
    {}

    AmpInfoTableImpl(AmpInfoTableImpl const & other) : AmpInfoTable(other) {}

private:

    virtual PTR(BaseTable) _clone() const {
        return boost::make_shared<AmpInfoTableImpl>(*this);
    }

    virtual PTR(BaseRecord) _makeRecord() {
        PTR(AmpInfoRecord) record = boost::make_shared<AmpInfoRecordImpl>(getSelf<AmpInfoTableImpl>());
        if (getIdFactory()) record->setId((*getIdFactory())());
        return record;
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
    PTR(AmpInfoTable) table =  AmpInfoTable::make(schema, PTR(IdFactory)());
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

PTR(AmpInfoTable) AmpInfoTable::make(Schema const & schema, PTR(IdFactory) const & idFactory) {
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            "Schema for AmpInfo must contain at least the keys defined by makeMinimalSchema()."
        );
    }
    return boost::make_shared<AmpInfoTableImpl>(schema, idFactory);
}

AmpInfoTable::AmpInfoTable(Schema const & schema, PTR(IdFactory) const & idFactory) :
    BaseTable(schema), _idFactory(idFactory) {}

AmpInfoTable::AmpInfoTable(AmpInfoTable const & other) :
    BaseTable(other), _idFactory(other._idFactory ? other._idFactory->clone() : other._idFactory) {}

AmpInfoTable::MinimalSchema::MinimalSchema() {
    id = schema.addField<RecordId>("id", "unique ID");
    name = schema.addField<std::string>("name", "name of amplifier location in camera");
    gain = schema.addField<double>("gain", "amplifier gain in e-/ADU", "e-/ADU");
    readnoise = schema.addField<double>("readnoise", "amplifier read noise, in e-", "e-");

    linearitycoeffs = schema.addField< Array<double> >("linearitycoeffs", "coefficients for linearity fit up to cubic", 4);
    linearitytype = schema.addField<std::string>("linearitytype", "type of linearity model");

    trimmedbbox_ll = schema.addField< Point<int> >("trimmedbbox_ll", "LL pixel of amplifier pixels in assembled image", "pixels");
    trimmedbbox_ur = schema.addField< Point<int> >("trimmedbbox_ur", "UR pixel of amplifier pixels in assembled image", "pixels");
    // Raw data fields
    hasrawamplifier = schema.addField<Flag>("hasrawamplifier", 
                      "does the amp have raw information (e.g. untrimmed bounding boxes)");

    flipx = schema.addField<Flag>("flipx", "flip row order to make assembled image?");
    flipy = schema.addField<Flag>("flipy", "flip column order to make an assembled image?");

    rawxyoffset = schema.addField< Point<int> >("rawxyoffset", 
                  "offset for assembling a raw CCD image: desired xy0 - raw xy0", "pixels");

    rawbbox_ll = schema.addField< Point<int> >("rawbbox_ll", "LL pixel of all amplifier pixels on raw image", "pixels");
    rawbbox_ur = schema.addField< Point<int> >("rawbbox_ur", "UR pixel bounding box of all amplifier pixels on raw image", "pixels");
    databbox_ll = schema.addField< Point<int> >("databbox_ll", "LL pixel of amplifier image pixels on raw image", "pixels");
    databbox_ur = schema.addField< Point<int> >("databbox_ur", "UR pixel of amplifier image pixels on raw image", "pixels");
    horizontaloverscanbbox_ll = schema.addField< Point<int> >("horizontaloverscanbbox_ll", 
                             "LL pixel of usable horizontal overscan pixels", "pixels");
    horizontaloverscanbbox_ur = schema.addField< Point<int> >("horizontaloverscanbboxUur", 
                             "UR pixel of usable horizontal overscan pixels", "pixels");
    verticaloverscanbbox_ll = schema.addField< Point<int> >("verticaloverscanbbox_ll", 
                             "LL pixel of usable vertical overscan pixels", "pixels");
    verticaloverscanbbox_ur = schema.addField< Point<int> >("verticaloverscanbbox_ur", 
                             "UR pixel of usable vertical overscan pixels", "pixels");
    prescanbbox_ll = schema.addField< Point<int> >("prescanbbox_ll",
                             "LL pixel of usable (horizontal) prescan pixels on raw image", "pixels");
    prescanbbox_ur = schema.addField< Point<int> >("prescanbbox_ll",
                             "UR pixel of usable (horizontal) prescan pixels on raw image", "pixels");
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
RecordId AmpInfoRecord::getId() const { return get(AmpInfoTable::getIdKey()); }
void AmpInfoRecord::setId(RecordId id) { set(AmpInfoTable::getIdKey(), id); }

std::string AmpInfoRecord::getName() const { return get(AmpInfoTable::getNameKey()); }
void AmpInfoRecord::setName(std::string const &name) { set(AmpInfoTable::getNameKey(), name); }

geom::Box2I AmpInfoRecord::getTrimmedBbox() const { 
    return geom::Box2I(get(AmpInfoTable::getTrimmedBboxLLKey()), get(AmpInfoTable::getTrimmedBboxURKey())); 
}

void AmpInfoRecord::setTrimmedBbox(geom::Box2I const &trimmedbbox) { 
    set(AmpInfoTable::getTrimmedBboxLLKey(), trimmedbbox.getMin()); 
    set(AmpInfoTable::getTrimmedBboxURKey(), trimmedbbox.getMax()); 
}

double AmpInfoRecord::getGain() const { return get(AmpInfoTable::getGainKey()); }
void AmpInfoRecord::setGain(double gain) { set(AmpInfoTable::getGainKey(), gain); }

double AmpInfoRecord::getReadNoise() const { return get(AmpInfoTable::getReadNoiseKey()); }
void AmpInfoRecord::setReadNoise(double readnoise) { set(AmpInfoTable::getReadNoiseKey(), readnoise); }

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

bool AmpInfoRecord::getHasRawAmplifier() const { return get(AmpInfoTable::getHasRawAmplifierKey()); }
void AmpInfoRecord::setHasRawAmplifier(bool hasrawamplifier) { set(AmpInfoTable::getHasRawAmplifierKey(), hasrawamplifier); }

geom::Box2I AmpInfoRecord::getRawBbox() const { 
    return geom::Box2I(get(AmpInfoTable::getRawBboxLLKey()), get(AmpInfoTable::getRawBboxURKey())); 
}

void AmpInfoRecord::setRawBbox(geom::Box2I const &trimmedbbox) { 
    set(AmpInfoTable::getRawBboxLLKey(), trimmedbbox.getMin()); 
    set(AmpInfoTable::getRawBboxURKey(), trimmedbbox.getMax()); 
}

geom::Box2I AmpInfoRecord::getDataBbox() const { 
    return geom::Box2I(get(AmpInfoTable::getDataBboxLLKey()), get(AmpInfoTable::getDataBboxURKey())); 
}

void AmpInfoRecord::setDataBbox(geom::Box2I const &trimmedbbox) { 
    set(AmpInfoTable::getDataBboxLLKey(), trimmedbbox.getMin()); 
    set(AmpInfoTable::getDataBboxURKey(), trimmedbbox.getMax()); 
}

geom::Box2I AmpInfoRecord::getHorizontalOverscanBbox() const { 
    return geom::Box2I(get(AmpInfoTable::getHorizontalOverscanBboxLLKey()), get(AmpInfoTable::getHorizontalOverscanBboxURKey())); 
}

void AmpInfoRecord::setHorizontalOverscanBbox(geom::Box2I const &trimmedbbox) { 
    set(AmpInfoTable::getHorizontalOverscanBboxLLKey(), trimmedbbox.getMin()); 
    set(AmpInfoTable::getHorizontalOverscanBboxURKey(), trimmedbbox.getMax()); 
}

geom::Box2I AmpInfoRecord::getVerticalOverscanBbox() const { 
    return geom::Box2I(get(AmpInfoTable::getVerticalOverscanBboxLLKey()), get(AmpInfoTable::getVerticalOverscanBboxURKey())); 
}

void AmpInfoRecord::setVerticalOverscanBbox(geom::Box2I const &trimmedbbox) { 
    set(AmpInfoTable::getVerticalOverscanBboxLLKey(), trimmedbbox.getMin()); 
    set(AmpInfoTable::getVerticalOverscanBboxURKey(), trimmedbbox.getMax()); 
}

geom::Box2I AmpInfoRecord::getPrescanBbox() const { 
    return geom::Box2I(get(AmpInfoTable::getPrescanBboxLLKey()), get(AmpInfoTable::getPrescanBboxURKey())); 
}

void AmpInfoRecord::setPrescanBbox(geom::Box2I const &trimmedbbox) { 
    set(AmpInfoTable::getPrescanBboxLLKey(), trimmedbbox.getMin()); 
    set(AmpInfoTable::getPrescanBboxURKey(), trimmedbbox.getMax()); 
}

bool AmpInfoRecord::getFlipX() const { return get(AmpInfoTable::getFlipXKey()); }
void AmpInfoRecord::setFlipX(bool flipx) { set(AmpInfoTable::getFlipXKey(), flipx); }

bool AmpInfoRecord::getFlipY() const { return get(AmpInfoTable::getFlipYKey()); }
void AmpInfoRecord::setFlipY(bool flipy) { set(AmpInfoTable::getFlipYKey(), flipy); }

geom::Extent2I AmpInfoRecord::getRawXYOffset() const { return geom::Extent2I(get(AmpInfoTable::getRawXYOffsetKey())); }
void AmpInfoRecord::setRawXYOffset(geom::Extent2I const &rawxyoffset) { 
    set(AmpInfoTable::getRawXYOffsetKey(), geom::Point2I(rawxyoffset.asPair())); 
}

template class CatalogT<AmpInfoRecord>;
template class CatalogT<AmpInfoRecord const>;

template class SortedCatalogT<AmpInfoRecord>;
template class SortedCatalogT<AmpInfoRecord const>;

}}} // namespace lsst::afw::table
