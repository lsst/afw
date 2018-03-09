// -*- lsst-c++ -*-
#include <typeinfo>

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/io/FitsReader.h"
#include "lsst/afw/table/Simple.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst {
namespace afw {
namespace table {

//-----------------------------------------------------------------------------------------------------------
//----- SimpleFitsWriter ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsWriter for Simple - this just sets the AFW_TYPE key to SIMPLE, which should ensure
// we use SimpleFitsReader to read it.

namespace {

class SimpleFitsWriter : public io::FitsWriter {
public:
    explicit SimpleFitsWriter(Fits* fits, int flags) : io::FitsWriter(fits, flags) {}

protected:
    void _writeTable(std::shared_ptr<BaseTable const> const& table, std::size_t nRows) override;
};

void SimpleFitsWriter::_writeTable(std::shared_ptr<BaseTable const> const& t, std::size_t nRows) {
    std::shared_ptr<SimpleTable const> table = std::dynamic_pointer_cast<SimpleTable const>(t);
    if (!table) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,
                          "Cannot use a SimpleFitsWriter on a non-Simple table.");
    }
    io::FitsWriter::_writeTable(table, nRows);
    _fits->writeKey("AFW_TYPE", "SIMPLE", "Tells lsst::afw to load this as a Simple table.");
}

}  // namespace

//-----------------------------------------------------------------------------------------------------------
//----- SimpleFitsReader ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsReader for SimpleTable/Record - this gets registered with name SIMPLE, so it should get used
// whenever we read a table with AFW_TYPE set to that value.

namespace {

class SimpleFitsReader : public io::FitsReader {
public:
    SimpleFitsReader() : io::FitsReader("SIMPLE") {}

    std::shared_ptr<BaseTable> makeTable(io::FitsSchemaInputMapper& mapper,
                                                 std::shared_ptr<daf::base::PropertyList> metadata,
                                                 int ioFlags, bool stripMetadata) const override {
        std::shared_ptr<SimpleTable> table = SimpleTable::make(mapper.finalize());
        table->setMetadata(metadata);
        return table;
    }
};

// registers the reader so FitsReader::make can use it.
SimpleFitsReader const simpleFitsReader;

}  // namespace

//-----------------------------------------------------------------------------------------------------------
//----- SimpleTable/Record member function implementations -----------------------------------------------
//-----------------------------------------------------------------------------------------------------------

SimpleRecord::SimpleRecord(std::shared_ptr<SimpleTable> const& table) : BaseRecord(table) {}

SimpleRecord::~SimpleRecord() = default;

std::shared_ptr<SimpleTable> SimpleTable::make(Schema const& schema,
                                               std::shared_ptr<IdFactory> const& idFactory) {
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "Schema for Simple must contain at least the keys defined by makeMinimalSchema().");
    }
    return std::shared_ptr<SimpleTable>(new SimpleTable(schema, idFactory));
}

SimpleTable::SimpleTable(Schema const& schema, std::shared_ptr<IdFactory> const& idFactory)
        : BaseTable(schema), _idFactory(idFactory) {}

SimpleTable::SimpleTable(SimpleTable const& other)
        : BaseTable(other), _idFactory(other._idFactory ? other._idFactory->clone() : other._idFactory) {}
// Delegate to copy constructor for backwards compatibility
SimpleTable::SimpleTable(SimpleTable && other) : SimpleTable(other) {}

SimpleTable::~SimpleTable() = default;

SimpleTable::MinimalSchema::MinimalSchema() {
    id = schema.addField<RecordId>("id", "unique ID");
    coord = CoordKey::addFields(schema, "coord", "position in ra/dec");
    schema.getCitizen().markPersistent();
}

SimpleTable::MinimalSchema& SimpleTable::getMinimalSchema() {
    static MinimalSchema it;
    return it;
}

std::shared_ptr<io::FitsWriter> SimpleTable::makeFitsWriter(fits::Fits* fitsfile, int flags) const {
    return std::make_shared<SimpleFitsWriter>(fitsfile, flags);
}

std::shared_ptr<BaseTable> SimpleTable::_clone() const {
    return std::shared_ptr<SimpleTable>(new SimpleTable(*this));
}

std::shared_ptr<BaseRecord> SimpleTable::_makeRecord() {
    std::shared_ptr<SimpleRecord> record(new SimpleRecord(getSelf<SimpleTable>()));
    if (getIdFactory()) record->setId((*getIdFactory())());
    return record;
}

template class CatalogT<SimpleRecord>;
template class CatalogT<SimpleRecord const>;

template class SortedCatalogT<SimpleRecord>;
template class SortedCatalogT<SimpleRecord const>;
}  // namespace table
}  // namespace afw
}  // namespace lsst
