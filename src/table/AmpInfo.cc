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
    trimmedbbox = schema.addField<geom::Box2I>("trimmedbbox", "bounding box of amplifier pixels in assembled image", "pixels");
    gain = schema.addField<double>("gain", "amplifier gain in e-/ADU", "e-/ADU");
    readnoise = schema.addField<double>("readnoise", "amplifier read noise, in e-", "e-");
    linearitycoeffs = schema.addField<std::vector<double> >("linearitycoeffs", "coefficients for linearity fit");
    linearitytype = schema.addField<std::string>("linearitytype", "type of linearity model");
    hasrawamplifier = schema.addField<bool>("hasrawamplifier", 
                      "does the amp have raw information (e.g. untrimmed bounding boxes)");
    rawbbox = schema.addField<geom::Box2I>("rawbbox", "bounding box of all amplifier pixels on raw image", "pixels");
    databbox = schema.addField<geom::Box2I>("databbox", "bounding box of amplifier image pixels on raw image", "pixels");
    horizontaloverscanbbox = schema.addField<geom::Box2I>("horizontaloverscanbbox", 
                             "bounding box of usable horizontal overscan pixels", "pixels");
    verticaloverscanbbox = schema.addField<geom::Box2I>("verticaloverscanbbox", 
                             "bounding box of usable vertical overscan pixels", "pixels");
    prescanbbox = schema.addField<geom::Box2I>("prescanbbox",
                             "bounding box of usable (horizontal) prescan pixels on raw image", "pixels");
    flipx = schema.addField<bool>("flipx", "flip row order to make assembled image?");
    flipy = schema.addField<bool>("flipy", "flip column order to make an assembled image?");
    rawxyoffset = schema.addField<geom::Extent2I>("rawxyoffset", 
                  "offset for assembling a raw CCD image: desired xy0 - raw xy0", "pixels");
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

template class CatalogT<AmpInfoRecord>;
template class CatalogT<AmpInfoRecord const>;

template class SortedCatalogT<AmpInfoRecord>;
template class SortedCatalogT<AmpInfoRecord const>;

}}} // namespace lsst::afw::table
