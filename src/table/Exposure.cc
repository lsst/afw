// -*- lsst-c++ -*-
#include <typeinfo>

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/Exposure.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/afw/image/ApCorrMap.h"
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
    int version;
    Key<int> wcs;
    Key<int> psf;
    Key<int> calib;
    Key<int> apCorrMap;

    static PersistenceSchema const & get(int version=1) {
        static PersistenceSchema const instance0(0);
        static PersistenceSchema const instance1(1);
        switch (version) {
        case 0:
            return instance0;
        case 1:
            return instance1;
        default:
            assert(false);
        }
    }

    static PersistenceSchema const & getMatching(Schema const & schema) {
        if (schema.contains(PersistenceSchema::get(1).schema)) {
            return PersistenceSchema::get(1);
        } else if (schema.contains(PersistenceSchema::get(0).schema)) {
            return PersistenceSchema::get(0);
        }
        throw LSST_EXCEPT(
            afw::table::io::PersistenceError,
            "Unrecognized schema for ExposureCatalog"
        );
    }

    // Create a SchemaMapper that maps an ExposureRecord to a BaseRecord with IDs for Psf and Wcs.
    SchemaMapper makeWriteMapper(Schema const & inputSchema) const {
        assert(version > 0); // should only be writing version 1 catalogs
        std::vector<Schema> inSchemas;
        inSchemas.push_back(PersistenceSchema::get().schema);
        inSchemas.push_back(inputSchema);
        return SchemaMapper::join(inSchemas).back(); // don't need front; it's an identity mapper
    }

    // Create a SchemaMapper that maps a BaseRecord with IDs for Psf and Wcs to an ExposureRecord
    SchemaMapper makeReadMapper(Schema const & inputSchema) const {
        if (!inputSchema.contains(schema)) {
            throw LSST_EXCEPT(
                pex::exceptions::LogicErrorException,
                "Cannot read ExposureCatalog instances written with an older version of the pipeline"
            );
        }
        return SchemaMapper::removeMinimalSchema(inputSchema, schema);
    }

    // Convert an ExposureRecord to a BaseRecord with IDs for Psf and Wcs.
    template <typename OutputArchiveIsh>
    void writeRecord(
        ExposureRecord const & input, BaseRecord & output,
        SchemaMapper const & mapper, OutputArchiveIsh & archive,
        bool permissive
    ) const {
        assert(version > 0); // should only be writing version 1 catalogs
        output.assign(input, mapper);
        output.set(psf, archive.put(input.getPsf(), permissive));
        output.set(wcs, archive.put(input.getWcs(), permissive));
        output.set(calib, archive.put(input.getCalib(), permissive));
        output.set(apCorrMap, archive.put(input.getApCorrMap(), permissive));
    }

    void readRecord(
        BaseRecord const & input, ExposureRecord & output,
        SchemaMapper const & mapper, io::InputArchive const & archive
    ) const {
        output.assign(input, mapper);
        output.setPsf(archive.get<detection::Psf>(input.get(psf)));
        output.setWcs(archive.get<image::Wcs>(input.get(wcs)));
        output.setCalib(archive.get<image::Calib>(input.get(calib)));
        if (version > 0) {
            output.setApCorrMap(archive.get<image::ApCorrMap>(input.get(apCorrMap)));
        }
    }

private:

    PersistenceSchema(int version_) :
        schema(),
        version(version_),
        wcs(schema.addField<int>("wcs", "archive ID for Wcs object")),
        psf(schema.addField<int>("psf", "archive ID for Psf object")),
        calib(schema.addField<int>("calib", "archive ID for Calib object"))
    {
        if (version > 0) {
            apCorrMap = schema.addField<int>("apCorrMap", "archive ID for ApCorrMap object");
        }
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
            lsst::pex::exceptions::LogicErrorException,
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
        io::FitsReader(fits, archive, flags), _archive(archive), _helper(NULL)
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
    PersistenceSchema const * _helper;
};

PTR(BaseTable) ExposureFitsReader::_readTable() {
    PTR(daf::base::PropertyList) metadata = boost::make_shared<daf::base::PropertyList>();
    _fits->readMetadata(*metadata, true);
    Schema schema(*metadata, true);
    _inTable = BaseTable::make(schema);
    _helper = &PersistenceSchema::getMatching(schema);
    _mapper = _helper->makeReadMapper(schema);
    PTR(ExposureTable) table = ExposureTable::make(_mapper.getOutputSchema());
    _startRecords(*table);
    if (metadata->exists("AFW_TYPE")) metadata->remove("AFW_TYPE");
    table->setMetadata(metadata);
    return table;
}

PTR(BaseRecord) ExposureFitsReader::_readRecord(PTR(BaseTable) const & t) {
    PTR(ExposureRecord) record;
    PTR(ExposureTable) table = boost::static_pointer_cast<ExposureTable>(t);
    PTR(BaseRecord) inRecord = io::FitsReader::_readRecord(_inTable);
    if (inRecord) {
        record = table->makeRecord();
        _helper->readRecord(*inRecord, *record, _mapper, *_archive);
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
            pex::exceptions::LogicErrorException,
            "ExposureRecord does not have a Wcs; cannot call contains()"
        );
    }
    try {
        geom::Point2D point = getWcs()->skyToPixel(coord);
        return geom::Box2D(getBBox()).contains(point);
    } catch (pex::exceptions::DomainErrorException &) {
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
        _apCorrMap = s._apCorrMap;
    } catch (std::bad_cast&) {}
}

PTR(ExposureTable) ExposureTable::make(Schema const & schema) {
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
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
    PersistenceSchema const & helper = PersistenceSchema::getMatching(catalog.getSchema());
    SchemaMapper mapper = helper.makeReadMapper(catalog.getSchema());
    ExposureCatalogT<ExposureRecord> result(mapper.getOutputSchema());
    result.reserve(catalog.size());
    for (BaseCatalog::const_iterator i = catalog.begin(); i != catalog.end(); ++i) {
        helper.readRecord(*i, *result.addNew(), mapper, archive);
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
