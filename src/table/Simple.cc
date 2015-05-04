// -*- lsst-c++ -*-
#include <typeinfo>

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/io/FitsReader.h"
#include "lsst/afw/table/Simple.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

//-----------------------------------------------------------------------------------------------------------
//----- Private SimpleTable/Record classes ---------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// These private derived classes are what you actually get when you do SimpleTable::make; like the
// private classes in BaseTable.cc, it's more convenient to have an extra set of trivial derived
// classes than to do a lot of friending.

namespace {

class SimpleTableImpl;

class SimpleRecordImpl : public SimpleRecord {
public:

    explicit SimpleRecordImpl(PTR(SimpleTable) const & table) : SimpleRecord(table) {}

};

class SimpleTableImpl : public SimpleTable {
public:

    explicit SimpleTableImpl(Schema const & schema, PTR(IdFactory) const & idFactory) :
        SimpleTable(schema, idFactory)
    {}

    SimpleTableImpl(SimpleTableImpl const & other) : SimpleTable(other) {}

private:

    virtual PTR(BaseTable) _clone() const {
        return boost::make_shared<SimpleTableImpl>(*this);
    }

    virtual PTR(BaseRecord) _makeRecord() {
        PTR(SimpleRecord) record = boost::make_shared<SimpleRecordImpl>(getSelf<SimpleTableImpl>());
        if (getIdFactory()) record->setId((*getIdFactory())());
        return record;
    }

};

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- SimpleFitsWriter ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsWriter for Simple - this just sets the AFW_TYPE key to SIMPLE, which should ensure
// we use SimpleFitsReader to read it.

namespace {

class SimpleFitsWriter : public io::FitsWriter {
public:

    explicit SimpleFitsWriter(Fits * fits, int flags) : io::FitsWriter(fits, flags) {}

protected:

    virtual void _writeTable(CONST_PTR(BaseTable) const & table, std::size_t nRows);

};

void SimpleFitsWriter::_writeTable(CONST_PTR(BaseTable) const & t, std::size_t nRows) {
    CONST_PTR(SimpleTable) table = boost::dynamic_pointer_cast<SimpleTable const>(t);
    if (!table) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicError,
            "Cannot use a SimpleFitsWriter on a non-Simple table."
        );
    }
    io::FitsWriter::_writeTable(table, nRows);
    _fits->writeKey("AFW_TYPE", "SIMPLE", "Tells lsst::afw to load this as a Simple table.");
}

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- SimpleFitsReader ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsReader for SimpleTable/Record - this gets registered with name SIMPLE, so it should get used
// whenever we read a table with AFW_TYPE set to that value.

namespace {

class SimpleFitsReader : public io::FitsReader {
public:

    SimpleFitsReader() : io::FitsReader("SIMPLE") {}

    virtual PTR(BaseTable) makeTable(
        io::FitsSchemaInputMapper & mapper,
        PTR(daf::base::PropertyList) metadata,
        int ioFlags,
        bool stripMetadata
    ) const {
        PTR(SimpleTable) table = SimpleTable::make(mapper.finalize());
        table->setMetadata(metadata);
        return table;
    }

};

// registers the reader so FitsReader::make can use it.
static SimpleFitsReader const simpleFitsReader;

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- SimpleTable/Record member function implementations -----------------------------------------------
//-----------------------------------------------------------------------------------------------------------

SimpleRecord::SimpleRecord(PTR(SimpleTable) const & table) : BaseRecord(table) {}

PTR(SimpleTable) SimpleTable::make(Schema const & schema, PTR(IdFactory) const & idFactory) {
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError,
            "Schema for Simple must contain at least the keys defined by makeMinimalSchema()."
        );
    }
    return boost::make_shared<SimpleTableImpl>(schema, idFactory);
}

SimpleTable::SimpleTable(Schema const & schema, PTR(IdFactory) const & idFactory) :
    BaseTable(schema), _idFactory(idFactory) {}

SimpleTable::SimpleTable(SimpleTable const & other) :
    BaseTable(other), _idFactory(other._idFactory ? other._idFactory->clone() : other._idFactory) {}

SimpleTable::MinimalSchema::MinimalSchema() {
    id = schema.addField<RecordId>("id", "unique ID");
    coord = CoordKey::addFields(schema, "coord", "position in ra/dec");
    schema.getCitizen().markPersistent();
}

SimpleTable::MinimalSchema & SimpleTable::getMinimalSchema() {
    static MinimalSchema it;
    return it;
}

PTR(io::FitsWriter)
SimpleTable::makeFitsWriter(fits::Fits * fitsfile, int flags) const {
    return boost::make_shared<SimpleFitsWriter>(fitsfile, flags);
}

template class CatalogT<SimpleRecord>;
template class CatalogT<SimpleRecord const>;

template class SortedCatalogT<SimpleRecord>;
template class SortedCatalogT<SimpleRecord const>;

}}} // namespace lsst::afw::table
