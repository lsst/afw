// -*- lsst-c++ -*-
#include <typeinfo>

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/Exposure.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/afw/detection/Psf.h"

namespace lsst { namespace afw { namespace table {

//-----------------------------------------------------------------------------------------------------------
//----- Private ExposureTable/Record classes ---------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// These private derived classes are what you actually get when you do ExposureTable::make; like the
// private classes in BaseTable.cc, it's more convenient to have an extra set of trivial derived
// classes than to do a lot of friending.

namespace {

class ExposureTableImpl;

class ExposureRecordImpl : public ExposureRecord {
public:

    explicit ExposureRecordImpl(PTR(ExposureTable) const & table) : ExposureRecord(table) {
        // Want to make default bbox empty, not a single pixel at 0,0
        this->set(ExposureTable::getBBoxMaxKey(), geom::Point2I(-1,-1));
    }

};

class ExposureTableImpl : public ExposureTable {
public:

    explicit ExposureTableImpl(Schema const & schema) : 
        ExposureTable(schema)
    {}

    ExposureTableImpl(ExposureTableImpl const & other) : ExposureTable(other) {}

private:

    virtual PTR(BaseTable) _clone() const {
        return boost::make_shared<ExposureTableImpl>(*this);
    }

    virtual PTR(BaseRecord) _makeRecord() {
        return boost::make_shared<ExposureRecordImpl>(getSelf<ExposureTableImpl>());
    }

};

// Schema prepended when saving an Exposure table
struct PersistenceSchema : private boost::noncopyable {
    Schema schema;
    Key<int> wcs;
    Key<int> psf;
    Key<int> calib;

    static PersistenceSchema const & get() {
        static PersistenceSchema const instance;
        return instance;
    }

    // Create a SchemaMapper that maps an ExposureRecord to a BaseRecord with IDs for Psf and Wcs.
    SchemaMapper makeWriteMapper(Schema const & inputSchema) const {
        std::vector<Schema> inSchemas;
        inSchemas.push_back(PersistenceSchema::get().schema);
        inSchemas.push_back(inputSchema);
        // don't need front; it's an identity mapper
        SchemaMapper result = SchemaMapper::join(inSchemas).back();
        result.editOutputSchema().setAliasMap(inputSchema.getAliasMap());
        return result;
    }

    // Create a SchemaMapper that maps a BaseRecord with IDs for Psf and Wcs to an ExposureRecord
    SchemaMapper makeReadMapper(Schema const & inputSchema) const {
        SchemaMapper result = SchemaMapper::removeMinimalSchema(inputSchema, schema);
        result.editOutputSchema().setAliasMap(inputSchema.getAliasMap());
        return result;
    }

    // Convert an ExposureRecord to a BaseRecord with IDs for Psf and Wcs.
    template <typename OutputArchiveIsh>
    void writeRecord(
        ExposureRecord const & input, BaseRecord & output,
        SchemaMapper const & mapper, OutputArchiveIsh & archive,
        bool permissive
    ) const {
        output.assign(input, mapper);
        output.set(psf, archive.put(input.getPsf(), permissive));
        output.set(wcs, archive.put(input.getWcs(), permissive));
        output.set(calib, archive.put(input.getCalib(), permissive));
    }

    void readRecord(
        BaseRecord const & input, ExposureRecord & output,
        SchemaMapper const & mapper, io::InputArchive const & archive
    ) const {
        output.assign(input, mapper);
        output.setPsf(archive.get<detection::Psf>(input.get(psf)));
        output.setWcs(archive.get<image::Wcs>(input.get(wcs)));
        output.setCalib(archive.get<image::Calib>(input.get(calib)));
    }

private:
    PersistenceSchema() :
        schema(),
        wcs(schema.addField<int>("wcs", "archive ID for Wcs object")),
        psf(schema.addField<int>("psf", "archive ID for Psf object")),
        calib(schema.addField<int>("calib", "archive ID for Calib object"))
    {
        schema.getCitizen().markPersistent();
    }
};

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- ExposureFitsWriter ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsWriter for Exposure - this just sets the AFW_TYPE key to EXPOSURE, which should ensure
// we use ExposureFitsReader to read it.

namespace {


class ExposureFitsWriter : public io::FitsWriter {
public:

    ExposureFitsWriter(Fits * fits, PTR(io::OutputArchive) archive, int flags)
        : io::FitsWriter(fits, flags), _doWriteArchive(false), _archive(archive)
    {
        if (!_archive) {
            _doWriteArchive = true;
            _archive.reset(new io::OutputArchive());
        }
    }

protected:
    
    virtual void _writeTable(CONST_PTR(BaseTable) const & table, std::size_t nRows);

    virtual void _writeRecord(BaseRecord const & r);

    virtual void _finish() {
        if (_doWriteArchive) _archive->writeFits(*_fits);
    }

    bool _doWriteArchive;
    PTR(io::OutputArchive) _archive;
    PTR(BaseRecord) _record;
    SchemaMapper _mapper;
};

void ExposureFitsWriter::_writeTable(CONST_PTR(BaseTable) const & t, std::size_t nRows) {
    CONST_PTR(ExposureTable) inTable = boost::dynamic_pointer_cast<ExposureTable const>(t);
    if (!inTable) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicError,
            "Cannot use a ExposureFitsWriter on a non-Exposure table."
        );
    }
    _mapper = PersistenceSchema::get().makeWriteMapper(inTable->getSchema());
    PTR(BaseTable) outTable = BaseTable::make(_mapper.getOutputSchema());
    io::FitsWriter::_writeTable(outTable, nRows);
    _fits->writeKey("AFW_TYPE", "EXPOSURE", "Tells lsst::afw to load this as an Exposure table.");
    _record = outTable->makeRecord();
}

void ExposureFitsWriter::_writeRecord(BaseRecord const & r) {
    ExposureRecord const & record = static_cast<ExposureRecord const &>(r);
    PersistenceSchema::get().writeRecord(record, *_record, _mapper, *_archive, false);
    io::FitsWriter::_writeRecord(*_record);
}

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- ExposureFitsReader ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsReader for ExposureTable/Record - this gets registered with name EXPOSURE, so it should get
// used whenever we read a table with AFW_TYPE set to that value.

namespace {

class ExposureFitsReader : public io::FitsReader {
public:

    explicit ExposureFitsReader(Fits * fits, PTR(io::InputArchive) archive, int flags) :
        io::FitsReader(fits, archive, flags), _archive(archive)
    {
        if (!_archive) {
            int oldHdu = _fits->getHdu();
            _fits->setHdu(oldHdu + 1);
            _archive.reset(new io::InputArchive(io::InputArchive::readFits(*_fits)));
            _fits->setHdu(oldHdu);
        }
    }

protected:

    virtual PTR(BaseTable) _readTable();

    virtual PTR(BaseRecord) _readRecord(PTR(BaseTable) const & table);

    PTR(BaseTable) _inTable;
    PTR(io::InputArchive) _archive;
    SchemaMapper _mapper;
};

PTR(BaseTable) ExposureFitsReader::_readTable() {
    PTR(daf::base::PropertyList) metadata = boost::make_shared<daf::base::PropertyList>();
    _fits->readMetadata(*metadata, true);
    Schema schema(*metadata, true);
    _inTable = BaseTable::make(schema);
    _mapper = PersistenceSchema::get().makeReadMapper(schema);
    PTR(ExposureTable) table = ExposureTable::make(_mapper.getOutputSchema());
    table->setMetadata(metadata);
    _startRecords(*table);
    return table;
}

PTR(BaseRecord) ExposureFitsReader::_readRecord(PTR(BaseTable) const & t) {
    PTR(ExposureRecord) record;
    PTR(ExposureTable) table = boost::static_pointer_cast<ExposureTable>(t);
    PTR(BaseRecord) inRecord = io::FitsReader::_readRecord(_inTable);
    if (inRecord) {
        record = table->makeRecord();
        PersistenceSchema::get().readRecord(*inRecord, *record, _mapper, *_archive);
    }
    return record;
}

// registers the reader so FitsReader::make can use it.
static io::FitsReader::FactoryT<ExposureFitsReader> referenceFitsReaderFactory("EXPOSURE");

} // anonymous

//-----------------------------------------------------------------------------------------------------------
//----- ExposureTable/Record member function implementations -----------------------------------------------
//-----------------------------------------------------------------------------------------------------------

geom::Box2I ExposureRecord::getBBox() const {
    return geom::Box2I(get(ExposureTable::getBBoxMinKey()), get(ExposureTable::getBBoxMaxKey()));
}

void ExposureRecord::setBBox(geom::Box2I const & bbox) {
    set(ExposureTable::getBBoxMinKey(), bbox.getMin());
    set(ExposureTable::getBBoxMaxKey(), bbox.getMax());
}

bool ExposureRecord::contains(Coord const & coord) const {
    if (!getWcs()) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicError,
            "ExposureRecord does not have a Wcs; cannot call contains()"
        );
    }
    try {
        geom::Point2D point = getWcs()->skyToPixel(coord);
        return geom::Box2D(getBBox()).contains(point);
    } catch (pex::exceptions::DomainError &) {
        // Wcs can throw if the given coordinate is outside the region
        // where the Wcs is valid.
        return false;
    }
}

bool ExposureRecord::contains(geom::Point2D const & point, image::Wcs const & wcs) const {
    return contains(*wcs.pixelToSky(point));
}

ExposureRecord::ExposureRecord(PTR(ExposureTable) const & table) : BaseRecord(table) {}

void ExposureRecord::_assign(BaseRecord const & other) {
    try {
        ExposureRecord const & s = dynamic_cast<ExposureRecord const &>(other);
        _psf = s._psf;
        _wcs = s._wcs;
        _calib = s._calib;
    } catch (std::bad_cast&) {}
}

PTR(ExposureTable) ExposureTable::make(Schema const & schema) {
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError,
            "Schema for Exposure must contain at least the keys defined by makeMinimalSchema()."
        );
    }
    return boost::make_shared<ExposureTableImpl>(schema);
}

ExposureTable::ExposureTable(Schema const & schema) :
    BaseTable(schema) {}

ExposureTable::ExposureTable(ExposureTable const & other) :
    BaseTable(other) {}

ExposureTable::MinimalSchema::MinimalSchema() {
    id = schema.addField<RecordId>("id", "unique ID");
    bboxMin = schema.addField< Point<int> >("bbox.min", "bbox minimum point", "pixels");
    bboxMax = schema.addField< Point<int> >("bbox.max", "bbox maximum point", "pixels");
    schema.getCitizen().markPersistent();
}

ExposureTable::MinimalSchema & ExposureTable::getMinimalSchema() {
    static MinimalSchema it;
    return it;
}

PTR(io::FitsWriter)
ExposureTable::makeFitsWriter(fits::Fits * fitsfile, int flags) const {
    return boost::make_shared<ExposureFitsWriter>(fitsfile, PTR(io::OutputArchive)(), flags);
}

PTR(io::FitsWriter)
ExposureTable::makeFitsWriter(fits::Fits * fitsfile, PTR(io::OutputArchive) archive, int flags) const {
    return boost::make_shared<ExposureFitsWriter>(fitsfile, archive, flags);
}

//-----------------------------------------------------------------------------------------------------------
//----- ExposureCatalogT member function implementations ----------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

template <typename RecordT>
void ExposureCatalogT<RecordT>::writeToArchive(io::OutputArchiveHandle & handle, bool permissive) const {
    SchemaMapper mapper = PersistenceSchema::get().makeWriteMapper(this->getSchema());
    BaseCatalog outputCat = handle.makeCatalog(mapper.getOutputSchema());
    outputCat.reserve(this->size());
    for (const_iterator i = this->begin(); i != this->end(); ++i) {
        PersistenceSchema::get().writeRecord(*i, *outputCat.addNew(), mapper, handle, permissive);
    }
    handle.saveCatalog(outputCat);
}

template <typename RecordT>
ExposureCatalogT<RecordT> ExposureCatalogT<RecordT>::readFromArchive(
    io::InputArchive const & archive, BaseCatalog const & catalog
) {
    SchemaMapper mapper = PersistenceSchema::get().makeReadMapper(catalog.getSchema());
    ExposureCatalogT<ExposureRecord> result(mapper.getOutputSchema());
    result.reserve(catalog.size());
    for (BaseCatalog::const_iterator i = catalog.begin(); i != catalog.end(); ++i) {
        PersistenceSchema::get().readRecord(*i, *result.addNew(), mapper, archive);
    }
    return result;
}

template <typename RecordT>
ExposureCatalogT<RecordT>
ExposureCatalogT<RecordT>::subsetContaining(Coord const & coord) const {
    ExposureCatalogT result(this->getTable());
    for (const_iterator i = this->begin(); i != this->end(); ++i) {
        if (i->contains(coord)) result.push_back(i);
    }
    return result;
}

template <typename RecordT>
ExposureCatalogT<RecordT>
ExposureCatalogT<RecordT>::subsetContaining(geom::Point2D const & point, image::Wcs const & wcs) const {
    ExposureCatalogT result(this->getTable());
    for (const_iterator i = this->begin(); i != this->end(); ++i) {
        if (i->contains(point, wcs)) result.push_back(i);
    }
    return result;
}

//-----------------------------------------------------------------------------------------------------------
//----- Explicit instantiation ------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

template class CatalogT<ExposureRecord>;
template class CatalogT<ExposureRecord const>;

template class SortedCatalogT<ExposureRecord>;
template class SortedCatalogT<ExposureRecord const>;

template class ExposureCatalogT<ExposureRecord>;
template class ExposureCatalogT<ExposureRecord const>;

}}} // namespace lsst::afw::table
