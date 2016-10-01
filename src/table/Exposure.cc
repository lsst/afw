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
#include "lsst/afw/geom/polygon/Polygon.h"
#include "lsst/afw/image/VisitInfo.h"

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
        return std::make_shared<ExposureTableImpl>(*this);
    }

    virtual PTR(BaseRecord) _makeRecord() {
        return std::make_shared<ExposureRecordImpl>(getSelf<ExposureTableImpl>());
    }

};

// Schema prepended when saving an Exposure table
struct PersistenceSchema {
    Schema schema;
    Key<int> wcs;
    Key<int> psf;
    Key<int> calib;
    Key<int> apCorrMap;
    Key<int> validPolygon;
    Key<int> visitInfo;

    static PersistenceSchema const & get() {
        static PersistenceSchema const instance;
        return instance;
    }

    // Create a SchemaMapper that maps an ExposureRecord to a BaseRecord
    // with IDs for Wcs, Psf, Calib and ApCorrMap.
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
        output.set(apCorrMap, archive.put(input.getApCorrMap(), permissive));
        output.set(validPolygon, archive.put(input.getValidPolygon(), permissive));
        output.set(visitInfo, archive.put(input.getVisitInfo(), permissive));
    }

    void readRecord(
        BaseRecord const & input, ExposureRecord & output,
        SchemaMapper const & mapper, io::InputArchive const & archive
    ) const {
        output.assign(input, mapper);
        output.setPsf(archive.get<detection::Psf>(input.get(psf)));
        output.setWcs(archive.get<image::Wcs>(input.get(wcs)));
        output.setCalib(archive.get<image::Calib>(input.get(calib)));
        output.setApCorrMap(archive.get<image::ApCorrMap>(input.get(apCorrMap)));
        output.setValidPolygon(archive.get<geom::polygon::Polygon>(input.get(validPolygon)));
        output.setVisitInfo(archive.get<image::VisitInfo>(input.get(visitInfo)));
    }

    // No copying
    PersistenceSchema (const PersistenceSchema&) = delete;
    PersistenceSchema& operator=(const PersistenceSchema&) = delete;

    // No moving
    PersistenceSchema (PersistenceSchema&&) = delete;
    PersistenceSchema& operator=(PersistenceSchema&&) = delete;

private:
    PersistenceSchema() :
        schema(),
        wcs(schema.addField<int>("wcs", "archive ID for Wcs object")),
        psf(schema.addField<int>("psf", "archive ID for Psf object")),
        calib(schema.addField<int>("calib", "archive ID for Calib object")),
        apCorrMap(schema.addField<int>("apCorrMap", "archive ID for ApCorrMap object")),
        validPolygon(schema.addField<int>("validPolygon", "archive ID for Polygon object")),
        visitInfo(schema.addField<int>("visitInfo", "archive ID for VisitInfo object"))
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
    CONST_PTR(ExposureTable) inTable = std::dynamic_pointer_cast<ExposureTable const>(t);
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

// FitsColumnReader that reads a Persistable subclass T (Wcs, Psf, or Calib here) by using an int
// column to retrieve the object from an InputArchive and attach it to an ExposureRecord via
// the Setter member function pointer.
template <typename T, void (ExposureRecord::*Setter)(PTR(T const))>
class PersistableObjectColumnReader : public io::FitsColumnReader {
public:

    static void setup(
        std::string const & name,
        io::FitsSchemaInputMapper & mapper
    ) {
        auto item = mapper.find(name);
        if (item) {
            if (mapper.hasArchive()) {
                std::unique_ptr<io::FitsColumnReader> reader(
                    new PersistableObjectColumnReader(item->column)
                );
                mapper.customize(std::move(reader));
            }
            mapper.erase(item);
        }
    }

    PersistableObjectColumnReader(int column) : _column(column) {}

    virtual void readCell(
        BaseRecord & record,
        std::size_t row,
        fits::Fits & fits,
        PTR(io::InputArchive) const & archive
    ) const {
        int id = 0;
        fits.readTableScalar<int>(row, _column, id);
        PTR(T) value = archive->get<T>(id);
        (static_cast<ExposureRecord&>(record).*(Setter))(value);
    }

private:
    bool _noHeavy;
    int _column;
};

namespace {

class ExposureFitsReader : public io::FitsReader {
public:

    ExposureFitsReader() : afw::table::io::FitsReader("EXPOSURE") {}

    virtual PTR(BaseTable) makeTable(
        io::FitsSchemaInputMapper & mapper,
        PTR(daf::base::PropertyList) metadata,
        int ioFlags,
        bool stripMetadata
    ) const {
        PersistableObjectColumnReader<detection::Psf,&ExposureRecord::setPsf>::setup("psf", mapper);
        PersistableObjectColumnReader<image::Wcs,&ExposureRecord::setWcs>::setup("wcs", mapper);
        PersistableObjectColumnReader<image::Calib,&ExposureRecord::setCalib>::setup("calib", mapper);
        PersistableObjectColumnReader<image::ApCorrMap,&ExposureRecord::setApCorrMap>::setup(
            "apCorrMap", mapper);
        PersistableObjectColumnReader<geom::polygon::Polygon,&ExposureRecord::setValidPolygon>::setup(
            "validPolygon", mapper);
        PersistableObjectColumnReader<image::VisitInfo, &ExposureRecord::setVisitInfo>::setup(
            "visitInfo", mapper);
        PTR(ExposureTable) table = ExposureTable::make(mapper.finalize());
        table->setMetadata(metadata);
        return table;
    }

    virtual bool usesArchive(int ioFlags) const { return true; }

};

static ExposureFitsReader const exposureFitsReader;

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

bool ExposureRecord::contains(Coord const & coord, bool includeValidPolygon) const {
    if (!getWcs()) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicError,
            "ExposureRecord does not have a Wcs; cannot call contains()"
        );
    }

    // If there is no valid polygon set to false
    if (includeValidPolygon && !getValidPolygon()) {
        includeValidPolygon = false;
    }

    try {
        geom::Point2D point = getWcs()->skyToPixel(coord);
        if (includeValidPolygon) return (geom::Box2D(getBBox()).contains(point) &&
                                 getValidPolygon()->contains(point));
        else return geom::Box2D(getBBox()).contains(point);
    } catch (pex::exceptions::DomainError &) {
        // Wcs can throw if the given coordinate is outside the region
        // where the Wcs is valid.
        return false;
    }
}

bool ExposureRecord::contains(geom::Point2D const & point, image::Wcs const & wcs,
                              bool includeValidPolygon) const {
    return contains(*wcs.pixelToSky(point), includeValidPolygon);
}

ExposureRecord::ExposureRecord(PTR(ExposureTable) const & table) : BaseRecord(table) {}

void ExposureRecord::_assign(BaseRecord const & other) {
    try {
        ExposureRecord const & s = dynamic_cast<ExposureRecord const &>(other);
        _psf = s._psf;
        _wcs = s._wcs;
        _calib = s._calib;
        _apCorrMap = s._apCorrMap;
        _validPolygon = s._validPolygon;
        _visitInfo = s._visitInfo;
    } catch (std::bad_cast&) {}
}

PTR(ExposureTable) ExposureTable::make(Schema const & schema) {
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError,
            "Schema for Exposure must contain at least the keys defined by makeMinimalSchema()."
        );
    }
    return std::make_shared<ExposureTableImpl>(schema);
}

ExposureTable::ExposureTable(Schema const & schema) :
    BaseTable(schema) {}

ExposureTable::ExposureTable(ExposureTable const & other) :
    BaseTable(other) {}

ExposureTable::MinimalSchema::MinimalSchema() {
    id = schema.addField<RecordId>("id", "unique ID");
    bboxMin = PointKey<int>::addFields(schema, "bbox_min", "bbox minimum point", "pixel");
    bboxMax = PointKey<int>::addFields(schema, "bbox_max", "bbox maximum point", "pixel");
    schema.getCitizen().markPersistent();
}

ExposureTable::MinimalSchema & ExposureTable::getMinimalSchema() {
    static MinimalSchema it;
    return it;
}

PTR(io::FitsWriter)
ExposureTable::makeFitsWriter(fits::Fits * fitsfile, int flags) const {
    return std::make_shared<ExposureFitsWriter>(fitsfile, PTR(io::OutputArchive)(), flags);
}

PTR(io::FitsWriter)
ExposureTable::makeFitsWriter(fits::Fits * fitsfile, PTR(io::OutputArchive) archive, int flags) const {
    return std::make_shared<ExposureFitsWriter>(fitsfile, archive, flags);
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
ExposureCatalogT<RecordT>::subsetContaining(Coord const & coord, bool includeValidPolygon) const {
    ExposureCatalogT result(this->getTable());
    for (const_iterator i = this->begin(); i != this->end(); ++i) {
        if (i->contains(coord, includeValidPolygon)) {
            result.push_back(i);
        }
    }
    return result;
}

template <typename RecordT>
ExposureCatalogT<RecordT>
ExposureCatalogT<RecordT>::subsetContaining(
    geom::Point2D const & point, image::Wcs const & wcs, bool includeValidPolygon
) const {
    ExposureCatalogT result(this->getTable());
    for (const_iterator i = this->begin(); i != this->end(); ++i) {
        if (i->contains(point, wcs, includeValidPolygon)) {
            result.push_back(i);
        }
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
