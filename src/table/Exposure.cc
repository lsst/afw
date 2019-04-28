// -*- lsst-c++ -*-
#include <memory>
#include <typeinfo>
#include <string>

#include "lsst/daf/base/PropertySet.h"
#include "lsst/daf/base/PropertyList.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/Exposure.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/afw/image/PhotoCalib.h"
#include "lsst/afw/image/ApCorrMap.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/geom/polygon/Polygon.h"
#include "lsst/afw/image/VisitInfo.h"
#include "lsst/afw/image/TransmissionCurve.h"
#include "lsst/afw/cameraGeom/Detector.h"

namespace lsst {
namespace afw {
namespace table {

//-----------------------------------------------------------------------------------------------------------
//----- Private ExposureTable/Record classes ---------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// These private derived classes are what you actually get when you do ExposureTable::make; like the
// private classes in BaseTable.cc, it's more convenient to have an extra set of trivial derived
// classes than to do a lot of friending.

namespace {

int const EXPOSURE_TABLE_CURRENT_VERSION = 5;                   // current version of ExposureTable
std::string const EXPOSURE_TABLE_VERSION_KEY = "EXPTABLE_VER";  // FITS header key for ExposureTable version

// Field names used to store the archive IDs of different components (used in
// multiple places, so we define them here instead of as multiple string
// literals).
std::string const WCS_FIELD_NAME = "wcs";
std::string const PSF_FIELD_NAME = "psf";
std::string const CALIB_FIELD_NAME = "calib";  // to support the deprecated Calib in old files
std::string const PHOTOCALIB_FIELD_NAME = "photoCalib";
std::string const VISIT_INFO_FIELD_NAME = "visitInfo";
std::string const AP_CORR_MAP_FIELD_NAME = "apCorrMap";
std::string const VALID_POLYGON_FIELD_NAME = "validPolygon";
std::string const TRANSMISSION_CURVE_FIELD_NAME = "transmissionCurve";
std::string const DETECTOR_FIELD_NAME = "detector";

int getTableVersion(daf::base::PropertySet &metadata) {
    return metadata.exists(EXPOSURE_TABLE_VERSION_KEY) ? metadata.get<int>(EXPOSURE_TABLE_VERSION_KEY) : 1;
}

/**
 * @internal Helper class for for persisting ExposureRecord
 *
 * Contains keys for columns beyond BaseRecord, a schema mapper and and helper functions
 */
struct PersistenceHelper {
    Schema schema;
    Key<int> wcs;
    Key<int> psf;
    Key<int> calib;  // to support the deprecated Calib in old files (replaced with photoCalib)
    Key<int> photoCalib;
    Key<int> apCorrMap;
    Key<int> validPolygon;
    Key<int> visitInfo;
    Key<int> transmissionCurve;
    Key<int> detector;

    // Create a SchemaMapper that maps an ExposureRecord to a BaseRecord with IDs for Wcs, Psf, etc.
    SchemaMapper makeWriteMapper(Schema const &inputSchema) const {
        std::vector<Schema> inSchemas;
        inSchemas.push_back(PersistenceHelper().schema);
        inSchemas.push_back(inputSchema);
        // don't need front; it's an identity mapper
        SchemaMapper result = SchemaMapper::join(inSchemas).back();
        result.editOutputSchema().setAliasMap(inputSchema.getAliasMap());
        return result;
    }

    // Create a SchemaMapper that maps a BaseRecord to an ExposureRecord with IDs for WCS, Psf, etc.
    SchemaMapper makeReadMapper(Schema const &inputSchema) const {
        SchemaMapper result = SchemaMapper::removeMinimalSchema(inputSchema, schema);
        result.editOutputSchema().setAliasMap(inputSchema.getAliasMap());
        return result;
    }

    // Write psf, wcs, etc. from an ExposureRecord to an archive
    template <typename OutputArchiveIsh>
    void writeRecord(ExposureRecord const &input, BaseRecord &output, SchemaMapper const &mapper,
                     OutputArchiveIsh &archive, bool permissive) const {
        output.assign(input, mapper);
        output.set(psf, archive.put(input.getPsf(), permissive));
        output.set(wcs, archive.put(input.getWcs(), permissive));
        output.set(photoCalib, archive.put(input.getPhotoCalib(), permissive));
        output.set(apCorrMap, archive.put(input.getApCorrMap(), permissive));
        output.set(validPolygon, archive.put(input.getValidPolygon(), permissive));
        output.set(visitInfo, archive.put(input.getVisitInfo(), permissive));
        output.set(transmissionCurve, archive.put(input.getTransmissionCurve(), permissive));
        output.set(detector, archive.put(input.getDetector(), permissive));
    }

    // Read psf, wcs, etc. from an archive to an ExposureRecord
    void readRecord(BaseRecord const &input, ExposureRecord &output, SchemaMapper const &mapper,
                    io::InputArchive const &archive) const {
        output.assign(input, mapper);
        if (psf.isValid()) {
            output.setPsf(archive.get<detection::Psf>(input.get(psf)));
        }
        if (wcs.isValid()) {
            output.setWcs(archive.get<geom::SkyWcs>(input.get(wcs)));
        }
        if (calib.isValid()) {
            output.setPhotoCalib(archive.get<image::PhotoCalib>(input.get(calib)));
        }
        if (photoCalib.isValid()) {
            output.setPhotoCalib(archive.get<image::PhotoCalib>(input.get(photoCalib)));
        }
        if (apCorrMap.isValid()) {
            output.setApCorrMap(archive.get<image::ApCorrMap>(input.get(apCorrMap)));
        }
        if (validPolygon.isValid()) {
            output.setValidPolygon(archive.get<geom::polygon::Polygon>(input.get(validPolygon)));
        }
        if (visitInfo.isValid()) {
            output.setVisitInfo(archive.get<image::VisitInfo>(input.get(visitInfo)));
        }
        if (transmissionCurve.isValid()) {
            output.setTransmissionCurve(archive.get<image::TransmissionCurve>(input.get(transmissionCurve)));
        }
        if (detector.isValid()) {
            output.setDetector(archive.get<cameraGeom::Detector>(input.get(detector)));
        }
    }

    // No copying
    PersistenceHelper(const PersistenceHelper &) = delete;
    PersistenceHelper &operator=(const PersistenceHelper &) = delete;

    // No moving
    PersistenceHelper(PersistenceHelper &&) = delete;
    PersistenceHelper &operator=(PersistenceHelper &&) = delete;

    // Construct a PersistenceHelper using the most modern schema.
    PersistenceHelper()
            : schema(),
              wcs(schema.addField<int>(WCS_FIELD_NAME, "archive ID for Wcs object")),
              psf(schema.addField<int>(PSF_FIELD_NAME, "archive ID for Psf object")),
              photoCalib(schema.addField<int>(PHOTOCALIB_FIELD_NAME, "archive ID for PhotoCalib object")),
              apCorrMap(schema.addField<int>(AP_CORR_MAP_FIELD_NAME, "archive ID for ApCorrMap object")),
              validPolygon(schema.addField<int>(VALID_POLYGON_FIELD_NAME, "archive ID for Polygon object")),
              visitInfo(schema.addField<int>(VISIT_INFO_FIELD_NAME, "archive ID for VisitInfo object")),
              transmissionCurve(schema.addField<int>(TRANSMISSION_CURVE_FIELD_NAME,
                                                     "archive ID for TransmissionCurve object")),
              detector(schema.addField<int>(DETECTOR_FIELD_NAME, "archive ID for Detector object")) {}

    // Add a field to this->schema, saving its key in 'key', if and only if 'name' is a field in 'oldSchema'
    void addIfPresent(Schema const &oldSchema, Key<int> &key, std::string const &name) {
        try {
            auto item = oldSchema.find<int>(name);
            key = schema.addField(item.field);
        } catch (pex::exceptions::NotFoundError &) {
        }
    }

    // Construct a PersistenceHelper from a possibly old on-disk schema
    PersistenceHelper(Schema const &oldSchema) {
        addIfPresent(oldSchema, wcs, WCS_FIELD_NAME);
        addIfPresent(oldSchema, psf, PSF_FIELD_NAME);
        addIfPresent(oldSchema, calib, CALIB_FIELD_NAME);
        addIfPresent(oldSchema, photoCalib, PHOTOCALIB_FIELD_NAME);
        addIfPresent(oldSchema, apCorrMap, AP_CORR_MAP_FIELD_NAME);
        addIfPresent(oldSchema, validPolygon, VALID_POLYGON_FIELD_NAME);
        addIfPresent(oldSchema, visitInfo, VISIT_INFO_FIELD_NAME);
        addIfPresent(oldSchema, transmissionCurve, TRANSMISSION_CURVE_FIELD_NAME);
        addIfPresent(oldSchema, detector, DETECTOR_FIELD_NAME);
        assert(oldSchema.contains(schema));
    }
};

}  // namespace

//-----------------------------------------------------------------------------------------------------------
//----- ExposureFitsWriter ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsWriter for Exposure - this sets the AFW_TYPE key to EXPOSURE, which should ensure
// we use ExposureFitsReader to read it, and sets EXPOSURE_TABLE_VERSION_KEY to the current version:
// EXPOSURE_TABLE_CURRENT_VERSION

namespace {

class ExposureFitsWriter : public io::FitsWriter {
public:
    ExposureFitsWriter(Fits *fits, std::shared_ptr<io::OutputArchive> archive, int flags)
            : io::FitsWriter(fits, flags), _doWriteArchive(false), _archive(archive), _helper() {
        if (!_archive) {
            _doWriteArchive = true;
            _archive.reset(new io::OutputArchive());
        }
    }

protected:
    void _writeTable(std::shared_ptr<BaseTable const> const &table, std::size_t nRows) override;

    void _writeRecord(BaseRecord const &r) override;

    void _finish() override {
        if (_doWriteArchive) _archive->writeFits(*_fits);
    }

    bool _doWriteArchive;
    std::shared_ptr<io::OutputArchive> _archive;
    std::shared_ptr<BaseRecord> _record;
    PersistenceHelper _helper;
    SchemaMapper _mapper;
};

void ExposureFitsWriter::_writeTable(std::shared_ptr<BaseTable const> const &t, std::size_t nRows) {
    std::shared_ptr<ExposureTable const> inTable = std::dynamic_pointer_cast<ExposureTable const>(t);
    if (!inTable) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,
                          "Cannot use a ExposureFitsWriter on a non-Exposure table.");
    }
    _mapper = _helper.makeWriteMapper(inTable->getSchema());
    std::shared_ptr<BaseTable> outTable = BaseTable::make(_mapper.getOutputSchema());
    io::FitsWriter::_writeTable(outTable, nRows);
    _fits->writeKey("AFW_TYPE", "EXPOSURE", "Tells lsst::afw to load this as an Exposure table.");
    _fits->writeKey(EXPOSURE_TABLE_VERSION_KEY, EXPOSURE_TABLE_CURRENT_VERSION, "Exposure table version");
    _record = outTable->makeRecord();
}

void ExposureFitsWriter::_writeRecord(BaseRecord const &r) {
    ExposureRecord const &record = static_cast<ExposureRecord const &>(r);
    _helper.writeRecord(record, *_record, _mapper, *_archive, false);
    io::FitsWriter::_writeRecord(*_record);
}

}  // namespace

//-----------------------------------------------------------------------------------------------------------
//----- ExposureFitsReader ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// FitsColumnReader that reads a Persistable subclass T (Wcs, Psf, or PhotoCalib here) by using an int
// column to retrieve the object from an InputArchive and attach it to an ExposureRecord via
// the Setter member function pointer.
template <typename T, void (ExposureRecord::*Setter)(std::shared_ptr<T const>)>
class PersistableObjectColumnReader : public io::FitsColumnReader {
public:
    static void setup(std::string const &name, io::FitsSchemaInputMapper &mapper) {
        auto item = mapper.find(name);
        if (item) {
            if (mapper.hasArchive()) {
                std::unique_ptr<io::FitsColumnReader> reader(new PersistableObjectColumnReader(item->column));
                mapper.customize(std::move(reader));
            }
            mapper.erase(item);
        }
    }

    PersistableObjectColumnReader(int column) : _column(column) {}

    void readCell(BaseRecord &record, std::size_t row, fits::Fits &fits,
                  std::shared_ptr<io::InputArchive> const &archive) const override {
        int id = 0;
        fits.readTableScalar<int>(row, _column, id);
        std::shared_ptr<T> value = archive->get<T>(id);
        (static_cast<ExposureRecord &>(record).*(Setter))(value);
    }

private:
    bool _noHeavy;
    int _column;
};

namespace {

class ExposureFitsReader : public io::FitsReader {
public:
    ExposureFitsReader() : afw::table::io::FitsReader("EXPOSURE") {}

    std::shared_ptr<BaseTable> makeTable(io::FitsSchemaInputMapper &mapper,
                                         std::shared_ptr<daf::base::PropertyList> metadata, int ioFlags,
                                         bool stripMetadata) const override {
        // We rely on the table version stored in the metadata when loading an ExposureCatalog
        // persisted on its own.  This is not as flexible in terms of backwards compatibility
        // as the code that loads ExposureCatalogs persisted as part of something else, but
        // we happen to know there are no ExposureCatalogs sitting on disk with with versions
        // older than what this routine supports.
        auto tableVersion = getTableVersion(*metadata);
        PersistableObjectColumnReader<detection::Psf, &ExposureRecord::setPsf>::setup("psf", mapper);
        PersistableObjectColumnReader<geom::SkyWcs, &ExposureRecord::setWcs>::setup("wcs", mapper);
        PersistableObjectColumnReader<image::ApCorrMap, &ExposureRecord::setApCorrMap>::setup("apCorrMap",
                                                                                              mapper);
        PersistableObjectColumnReader<geom::polygon::Polygon, &ExposureRecord::setValidPolygon>::setup(
                "validPolygon", mapper);
        if (tableVersion > 1) {
            PersistableObjectColumnReader<image::VisitInfo, &ExposureRecord::setVisitInfo>::setup("visitInfo",
                                                                                                  mapper);
        }
        if (tableVersion > 2) {
            PersistableObjectColumnReader<image::TransmissionCurve,
                                          &ExposureRecord::setTransmissionCurve>::setup("transmissionCurve",
                                                                                        mapper);
        }
        if (tableVersion > 3) {
            PersistableObjectColumnReader<cameraGeom::Detector, &ExposureRecord::setDetector>::setup(
                    "detector", mapper);
        }
        // Load the PhotoCalib from the `calib` table prior to version 5.
        if (tableVersion <= 4) {
            PersistableObjectColumnReader<image::PhotoCalib, &ExposureRecord::setPhotoCalib>::setup("calib",
                                                                                                    mapper);
        } else {
            PersistableObjectColumnReader<image::PhotoCalib, &ExposureRecord::setPhotoCalib>::setup(
                    "photoCalib", mapper);
        }

        auto schema = mapper.finalize();
        std::shared_ptr<ExposureTable> table = ExposureTable::make(schema);
        table->setMetadata(metadata);
        return table;
    }

    bool usesArchive(int ioFlags) const override { return true; }
};

static ExposureFitsReader const exposureFitsReader;

}  // namespace

//-----------------------------------------------------------------------------------------------------------
//----- ExposureTable/Record member function implementations -----------------------------------------------
//-----------------------------------------------------------------------------------------------------------

lsst::geom::Box2I ExposureRecord::getBBox() const {
    return lsst::geom::Box2I(get(ExposureTable::getBBoxKey()));
}

void ExposureRecord::setBBox(lsst::geom::Box2I const &bbox) { set(ExposureTable::getBBoxKey(), bbox); }

bool ExposureRecord::contains(lsst::geom::SpherePoint const &coord, bool includeValidPolygon) const {
    if (!getWcs()) {
        throw LSST_EXCEPT(pex::exceptions::LogicError,
                          "ExposureRecord does not have a Wcs; cannot call contains()");
    }

    // If there is no valid polygon set to false
    if (includeValidPolygon && !getValidPolygon()) {
        includeValidPolygon = false;
    }

    try {
        lsst::geom::Point2D point = getWcs()->skyToPixel(coord);
        if (includeValidPolygon)
            return (lsst::geom::Box2D(getBBox()).contains(point) && getValidPolygon()->contains(point));
        else
            return lsst::geom::Box2D(getBBox()).contains(point);
    } catch (pex::exceptions::DomainError &) {
        // SkyWcs can throw if the given coordinate is outside the region where the WCS is valid.
        return false;
    }
}

bool ExposureRecord::contains(lsst::geom::Point2D const &point, geom::SkyWcs const &wcs,
                              bool includeValidPolygon) const {
    return contains(wcs.pixelToSky(point), includeValidPolygon);
}

ExposureRecord::~ExposureRecord() = default;

void ExposureRecord::_assign(BaseRecord const &other) {
    try {
        ExposureRecord const &s = dynamic_cast<ExposureRecord const &>(other);
        _psf = s._psf;
        _wcs = s._wcs;
        _photoCalib = s._photoCalib;
        _apCorrMap = s._apCorrMap;
        _validPolygon = s._validPolygon;
        _visitInfo = s._visitInfo;
        _transmissionCurve = s._transmissionCurve;
        _detector = s._detector;
    } catch (std::bad_cast &) {
    }
}

std::shared_ptr<ExposureTable> ExposureTable::make(Schema const &schema) {
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterError,
                "Schema for Exposure must contain at least the keys defined by makeMinimalSchema().");
    }
    return std::shared_ptr<ExposureTable>(new ExposureTable(schema));
}

ExposureTable::ExposureTable(Schema const &schema) : BaseTable(schema) {}

ExposureTable::ExposureTable(ExposureTable const &other) : BaseTable(other) {}
// Delegate to copy-constructor for backward compatibility
ExposureTable::ExposureTable(ExposureTable &&other) : ExposureTable(other) {}

ExposureTable::~ExposureTable() = default;

ExposureTable::MinimalSchema::MinimalSchema() {
    id = schema.addField<RecordId>("id", "unique ID");
    bbox = Box2IKey::addFields(schema, "bbox", "bounding box", "pixel");
    schema.getCitizen().markPersistent();
}

ExposureTable::MinimalSchema &ExposureTable::getMinimalSchema() {
    static MinimalSchema it;
    return it;
}

std::shared_ptr<io::FitsWriter> ExposureTable::makeFitsWriter(fits::Fits *fitsfile, int flags) const {
    return std::make_shared<ExposureFitsWriter>(fitsfile, std::shared_ptr<io::OutputArchive>(), flags);
}

std::shared_ptr<io::FitsWriter> ExposureTable::makeFitsWriter(fits::Fits *fitsfile,
                                                              std::shared_ptr<io::OutputArchive> archive,
                                                              int flags) const {
    return std::make_shared<ExposureFitsWriter>(fitsfile, archive, flags);
}

std::shared_ptr<BaseTable> ExposureTable::_clone() const {
    return std::shared_ptr<ExposureTable>(new ExposureTable(*this));
}

std::shared_ptr<BaseRecord> ExposureTable::_makeRecord() {
    return constructRecord<ExposureRecord>();
}

//-----------------------------------------------------------------------------------------------------------
//----- ExposureCatalogT member function implementations ----------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

template <typename RecordT>
void ExposureCatalogT<RecordT>::writeToArchive(io::OutputArchiveHandle &handle, bool permissive) const {
    PersistenceHelper helper{};
    SchemaMapper mapper = helper.makeWriteMapper(this->getSchema());
    BaseCatalog outputCat = handle.makeCatalog(mapper.getOutputSchema());
    outputCat.reserve(this->size());
    for (const_iterator i = this->begin(); i != this->end(); ++i) {
        helper.writeRecord(*i, *outputCat.addNew(), mapper, handle, permissive);
    }
    handle.saveCatalog(outputCat);
}

template <typename RecordT>
ExposureCatalogT<RecordT> ExposureCatalogT<RecordT>::readFromArchive(io::InputArchive const &archive,
                                                                     BaseCatalog const &catalog) {
    // Helper constructor will infer which components are available
    // (effectively the version, but more flexible).
    PersistenceHelper helper{catalog.getSchema()};
    SchemaMapper mapper = helper.makeReadMapper(catalog.getSchema());
    ExposureCatalogT<ExposureRecord> result(mapper.getOutputSchema());
    result.reserve(catalog.size());
    for (BaseCatalog::const_iterator i = catalog.begin(); i != catalog.end(); ++i) {
        helper.readRecord(*i, *result.addNew(), mapper, archive);
    }
    return result;
}

template <typename RecordT>
ExposureCatalogT<RecordT> ExposureCatalogT<RecordT>::subsetContaining(lsst::geom::SpherePoint const &coord,
                                                                      bool includeValidPolygon) const {
    ExposureCatalogT result(this->getTable());
    for (const_iterator i = this->begin(); i != this->end(); ++i) {
        if (i->contains(coord, includeValidPolygon)) {
            result.push_back(i);
        }
    }
    return result;
}

template <typename RecordT>
ExposureCatalogT<RecordT> ExposureCatalogT<RecordT>::subsetContaining(lsst::geom::Point2D const &point,
                                                                      geom::SkyWcs const &wcs,
                                                                      bool includeValidPolygon) const {
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
}  // namespace table
}  // namespace afw
}  // namespace lsst
