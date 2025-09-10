// -*- lsst-c++ -*-
#include <typeinfo>

#include "boost/format.hpp"

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/afw/detection/FootprintCtrl.h"
#include "lsst/afw/detection/HeavyFootprint.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"

namespace lsst {
namespace afw {
namespace table {

//-----------------------------------------------------------------------------------------------------------
//----- PersistenceHelpers ----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

namespace {

struct PersistenceHelper {
    SchemaMapper mapper;
    afw::table::Key<int> footprintKey;
};

}  // namespace

//-----------------------------------------------------------------------------------------------------------
//----- SourceFitsWriter ------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsWriter for Sources.  It also sets the AFW_TYPE key to SOURCE, which should
// ensure we use SourceFitsReader to read it.

// Because it also holds Footprints, a SourceCatalog isn't persisted with the same Schema
// it has in-memory; instead, it's saved in a Schema that has an additional int field
// appended to the end, which contains the afw::table::io "archive ID" that's used to
// extract the Footprint from additional FITS HDUs.  (If we disable saving Footprints via
// SourceFitsFlags, we do save using the original Schema).

// The only public access point to this class is SourceTable::makeFitsWriter.  If we
// subclass SourceTable someday, it may be necessary to put SourceFitsWriter in a header
// file so we can subclass it too.

namespace {

class SourceFitsWriter : public io::FitsWriter {
public:
    explicit SourceFitsWriter(Fits *fits, int flags) : io::FitsWriter(fits, flags) {}

protected:
    void _writeTable(std::shared_ptr<BaseTable const> const &table, std::size_t nRows) override;

    void _writeRecord(BaseRecord const &record) override;

    void _finish() override {
        if (!(_flags & SOURCE_IO_NO_FOOTPRINTS)) {
            _archive.writeFits(*_fits);
        }
    }

private:
    SchemaMapper _mapper;
    std::shared_ptr<BaseRecord> _outRecord;
    std::shared_ptr<BaseTable> _outTable;
    Key<int> _footprintKey;
    io::OutputArchive _archive;
};

void SourceFitsWriter::_writeTable(std::shared_ptr<BaseTable const> const &t, std::size_t nRows) {
    std::shared_ptr<SourceTable const> table = std::dynamic_pointer_cast<SourceTable const>(t);
    if (!table) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,
                          "Cannot use a SourceFitsWriter on a non-Source table.");
    }
    if (!(_flags & SOURCE_IO_NO_FOOTPRINTS)) {
        _mapper = SchemaMapper(t->getSchema(), true);
        _mapper.addMinimalSchema(t->getSchema(), true);
        _footprintKey = _mapper.editOutputSchema().addField<int>("footprint", "archive ID for Footprint");
        _outTable = BaseTable::make(_mapper.getOutputSchema());
        std::shared_ptr<daf::base::PropertyList> metadata = table->getMetadata();
        if (metadata) {
            metadata = std::static_pointer_cast<daf::base::PropertyList>(metadata->deepCopy());
        } else {
            metadata.reset(new daf::base::PropertyList());
        }
        // HDU 0 is empty (primary HDU can't be a table)
        // HDU 1 is the SourceCatalog's records
        // HDU 2 is the index for the afw::table::io archive that holds more complex objects
        //
        // Historically the AR_HDU keyword was 1-indexed (see RFC-304), and to maintain file compatibility
        // this is still the case so we're setting AR_HDU to 3 == 2 + 1
        metadata->set("AR_HDU", 3,
                      "HDU (1-indexed) containing the archive index for non-record data (e.g. Footprints)");
        _outTable->setMetadata(metadata);
        _outRecord = _outTable->makeRecord();  // make temporary record to use as a workspace
        io::FitsWriter::_writeTable(_outTable, nRows);
    } else {
        io::FitsWriter::_writeTable(table, nRows);
    }
    _fits->writeKey("AFW_TYPE", "SOURCE", "Tells lsst::afw to load this as a Source table.");
}

void SourceFitsWriter::_writeRecord(BaseRecord const &r) {
    SourceRecord const &record = static_cast<SourceRecord const &>(r);
    if (!(_flags & SOURCE_IO_NO_FOOTPRINTS)) {
        _outRecord->assign(record, _mapper);
        std::shared_ptr<afw::detection::Footprint> footprint = record.getFootprint();
        if (footprint) {
            if ((_flags & SOURCE_IO_NO_HEAVY_FOOTPRINTS) && footprint->isHeavy()) {
                footprint.reset(new afw::detection::Footprint(*footprint));
            }
            int footprintArchiveId = _archive.put(footprint);
            _outRecord->set(_footprintKey, footprintArchiveId);
        } else {
            _outRecord->set(_footprintKey, _archive.put(nullptr));
        }
        io::FitsWriter::_writeRecord(*_outRecord);
    } else {
        io::FitsWriter::_writeRecord(record);
    }
}

}  // namespace

//-----------------------------------------------------------------------------------------------------------
//----- SourceFitsReader ------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A custom FitsReader for Sources - this reads footprints in addition to the regular fields.  It
// gets registered with name SOURCE, so it should get used whenever we read a table with AFW_TYPE
// set to that value.  (The actual SourceFitsReader class is a bit further down, after some helper
// classes.)

// As noted in the comments for SourceFitsWriter, we add a column for the Footprint archive ID when
// we save a SourceCatalog.

// Things are a bit more complicated than that when reading, because we also need to be able to read files
// saved with an older version of the pipeline, in which there were 2-5 additional columns, all variable-
// length arrays, holding the Spans, Peaks, and HeavyFootprint arrays.  Those are handled by explicit
// calls to the FITS I/O routines here.

// The only public access point to this class is through the registry.  If we subclass SourceTable
// someday, it may be necessary to put SourceFitsReader in a header file so we can subclass it too.

namespace {

// FitsColumnReader subclass for backwards-compatible Footprint reading from variable-length arrays
class OldSourceFootprintReader : public io::FitsColumnReader {
public:
    static int readSpecialColumn(io::FitsSchemaInputMapper &mapper, daf::base::PropertyList &metadata,
                                 bool stripMetadata, std::string const &name) {
        int column = metadata.get(name, 0);
        --column;  // switch from 1-indexed to 0-indexed convention
        if (column >= 0) {
            if (stripMetadata) {
                metadata.remove(name);
            }
            mapper.erase(name);
        }
        return column;
    }

    static void setup(io::FitsSchemaInputMapper &mapper, daf::base::PropertyList &metadata, int ioFlags,
                      bool stripMetadata) {
        std::unique_ptr<OldSourceFootprintReader> reader(new OldSourceFootprintReader());
        reader->_spanCol = readSpecialColumn(mapper, metadata, stripMetadata, "SPANCOL");
        reader->_peakCol = readSpecialColumn(mapper, metadata, stripMetadata, "PEAKCOL");
        reader->_heavyPixCol = readSpecialColumn(mapper, metadata, stripMetadata, "HVYPIXCO");
        reader->_heavyMaskCol = readSpecialColumn(mapper, metadata, stripMetadata, "HVYMSKCO");
        reader->_heavyVarCol = readSpecialColumn(mapper, metadata, stripMetadata, "HVYVARCO");
        if ((ioFlags & SOURCE_IO_NO_FOOTPRINTS) || mapper.hasArchive()) {
            return;  // don't want to load anything, so we're done after just removing the special columns
        }
        if (ioFlags & SOURCE_IO_NO_HEAVY_FOOTPRINTS) {
            reader->_heavyPixCol = -1;
            reader->_heavyMaskCol = -1;
            reader->_heavyVarCol = -1;
        }
        // These checks are really basically assertions - they should only happen if we get
        // a corrupted catalog - but we still don't want to crash if that happens.
        if ((reader->_spanCol >= 0) != (reader->_peakCol >= 0)) {
            throw LSST_EXCEPT(afw::fits::FitsError,
                              "Corrupted catalog: either both or none of the Footprint Span/Peak columns "
                              "must be present.");
        }
        if (reader->_spanCol < 0) {
            return;
        }
        if ((reader->_heavyPixCol >= 0) != (reader->_heavyMaskCol >= 0) ||
            (reader->_heavyPixCol >= 0) != (reader->_heavyVarCol >= 0)) {
            throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    "Corrupted catalog: either all or none of the HeavyFootprint columns must be present.");
        }
        if (reader->_heavyPixCol >= 0 && reader->_spanCol < 0) {
            throw LSST_EXCEPT(afw::fits::FitsError,
                              "Corrupted catalog: HeavyFootprint columns with no Span/Peak columns.");
        }
        // If we do want to load old-style Footprints, add the column reader to the mapper.
        mapper.customize(std::move(reader));
    }

    void readCell(BaseRecord &baseRecord, std::size_t row, fits::Fits &fits,
                  std::shared_ptr<io::InputArchive> const &archive) const override {
        SourceRecord &record = static_cast<SourceRecord &>(baseRecord);
        std::vector<geom::Span> spansVector;

        // Load a regular Footprint from the span and peak columns.
        std::size_t spanElementCount = fits.getTableArraySize(row, _spanCol);
        std::size_t peakElementCount = fits.getTableArraySize(row, _peakCol);
        if (spanElementCount) {
            if (spanElementCount % 3) {
                throw LSST_EXCEPT(
                        afw::fits::FitsError,
                        afw::fits::makeErrorMessage(
                                fits.fptr, fits.status,
                                boost::format("Number of span elements (%d) must divisible by 3 (row %d)") %
                                        spanElementCount % row));
            }
            std::vector<int> spanElements(spanElementCount);
            fits.readTableArray(row, _spanCol, spanElementCount, &spanElements.front());
            std::vector<int>::iterator j = spanElements.begin();
            while (j != spanElements.end()) {
                int y = *j++;
                int x0 = *j++;
                int x1 = *j++;
                spansVector.emplace_back(y, x0, x1);
            }
        }
        std::shared_ptr<Footprint> fp = std::make_shared<detection::Footprint>(
                std::make_shared<geom::SpanSet>(std::move(spansVector)));
        if (peakElementCount) {
            if (peakElementCount % 3) {
                throw LSST_EXCEPT(
                        afw::fits::FitsError,
                        afw::fits::makeErrorMessage(
                                fits.fptr, fits.status,
                                boost::format("Number of peak elements (%d) must divisible by 3 (row %d)") %
                                        peakElementCount % row));
            }
            std::vector<float> peakElements(peakElementCount);
            fits.readTableArray(row, _peakCol, peakElementCount, &peakElements.front());
            std::vector<float>::iterator j = peakElements.begin();
            while (j != peakElements.end()) {
                float x = *j++;
                float y = *j++;
                float value = *j++;
                fp->addPeak(x, y, value);
            }
        }
        record.setFootprint(fp);

        // If we're setup to read HeavyFootprints
        if (_heavyPixCol < 0) {
            return;
        }
        std::size_t heavyPixElementCount = fits.getTableArraySize(row, _heavyPixCol);
        std::size_t heavyMaskElementCount = fits.getTableArraySize(row, _heavyMaskCol);
        std::size_t heavyVarElementCount = fits.getTableArraySize(row, _heavyVarCol);
        if (heavyPixElementCount > 0) {
            std::size_t N = fp->getArea();
            if ((heavyPixElementCount != N) || (heavyMaskElementCount != N) || (heavyVarElementCount != N)) {
                throw LSST_EXCEPT(
                        afw::fits::FitsError,
                        afw::fits::makeErrorMessage(
                                fits.fptr, fits.status,
                                boost::format("Number of HeavyFootprint elements (pix %d, mask %d, var %d) "
                                              "must all be equal to footprint area (%d)") %
                                        heavyPixElementCount % heavyMaskElementCount % heavyVarElementCount %
                                        N));
            }
            // float HeavyFootprints were the only kind we ever saved using the old format
            using HeavyFootprint = detection::HeavyFootprint<float>;
            std::shared_ptr<HeavyFootprint> heavy = std::make_shared<HeavyFootprint>(*fp);
            fits.readTableArray(row, _heavyPixCol, N, heavy->getImageArray().getData());
            fits.readTableArray(row, _heavyMaskCol, N, heavy->getMaskArray().getData());
            fits.readTableArray(row, _heavyVarCol, N, heavy->getVarianceArray().getData());
            record.setFootprint(heavy);
        }
    }

private:
    int _spanCol;
    int _peakCol;
    int _heavyPixCol;
    int _heavyMaskCol;
    int _heavyVarCol;
};

// FitsColumnReader for new-style Footprint persistence using archives.
class SourceFootprintReader : public io::FitsColumnReader {
public:
    static void setup(io::FitsSchemaInputMapper &mapper, int ioFlags) {
        auto item = mapper.find("footprint");
        if (item) {
            if (mapper.hasArchive()) {
                std::unique_ptr<io::FitsColumnReader> reader(
                        new SourceFootprintReader(ioFlags & SOURCE_IO_NO_HEAVY_FOOTPRINTS, item->column));
                mapper.customize(std::move(reader));
            }
            mapper.erase(item);
        }
    }

    SourceFootprintReader(bool noHeavy, int column) : _noHeavy(noHeavy), _column(column) {}

    void readCell(BaseRecord &record, std::size_t row, fits::Fits &fits,
                  std::shared_ptr<io::InputArchive> const &archive) const override {
        int id = 0;
        fits.readTableScalar<int>(row, _column, id);
        std::shared_ptr<Footprint> footprint = archive->get<Footprint>(id);
        if (_noHeavy && footprint->isHeavy()) {
            // It sort of defeats the purpose of the flag if we have to do the I/O to read
            // a HeavyFootprint before we can downgrade it to a regular Footprint, but that's
            // what we're going to do - at least this will save on on some memory usage, which
            // might still be useful.  It'd be really hard to fix this
            // (because we have no way to pass something like the ioFlags to the InputArchive).
            // The good news is that if someone's concerned about performance of reading
            // SourceCatalogs, they'll almost certainly use SOURCE_IO_NO_FOOTPRINTS, which
            // will do what we want.  SOURCE_IO_NO_HEAVY_FOOTPRINTS is more useful for writing
            // sources, and that still works just fine.
            footprint.reset(new Footprint(*footprint));
        }
        static_cast<SourceRecord &>(record).setFootprint(footprint);
    }

private:
    bool _noHeavy;
    int _column;
};

class SourceFitsReader : public io::FitsReader {
public:
    SourceFitsReader() : afw::table::io::FitsReader("SOURCE") {}

    std::shared_ptr<BaseTable> makeTable(io::FitsSchemaInputMapper &mapper,
                                         std::shared_ptr<daf::base::PropertyList> metadata, int ioFlags,
                                         bool stripMetadata) const override {
        // Look for old-style persistence of Footprints.  If we have both that and an archive, we
        // load the footprints from the archive, but still need to remove the old-style header keys
        // from the metadata and the corresponding fields from the FitsSchemaInputMapper.
        OldSourceFootprintReader::setup(mapper, *metadata, ioFlags, stripMetadata);
        // Look for new-style persistence of Footprints.  We'll only read them if we have an archive,
        // but we'll strip fields out regardless.
        SourceFootprintReader::setup(mapper, ioFlags);
        std::shared_ptr<SourceTable> table = SourceTable::make(mapper.finalize());
        table->setMetadata(metadata);
        return table;
    }

    bool usesArchive(int ioFlags) const override { return !(ioFlags & SOURCE_IO_NO_FOOTPRINTS); }
};

// registers the reader so FitsReader::make can use it.
static SourceFitsReader const sourceFitsReader;

}  // namespace

//-----------------------------------------------------------------------------------------------------------
//----- SourceTable/Record member function implementations --------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

SourceRecord::~SourceRecord() = default;

void SourceRecord::updateCoord(geom::SkyWcs const &wcs, bool include_covariance) {
    lsst::geom::Point2D center = getCentroid();
    setCoord(wcs.pixelToSky(center));
    if (include_covariance) {
        // Get coordinate covariance:
        auto err = getCentroidErr();
        CoordKey::ErrorKey const coordErrKey = CoordKey::getErrorKey(getSchema());
        Eigen::Matrix2f skyCov = calculateCoordCovariance(wcs, center, err);
        set(coordErrKey, skyCov);
    }
}

void SourceRecord::updateCoord(geom::SkyWcs const &wcs, PointKey<double> const &key, bool include_covariance) {
    lsst::geom::Point2D center = get(key);
    setCoord(wcs.pixelToSky(center));
    if (include_covariance) {
        // Get coordinate covariance:
        auto err = getCentroidErr();
        CoordKey::ErrorKey const coordErrKey = CoordKey::getErrorKey(getSchema());
        Eigen::Matrix2f skyCov = calculateCoordCovariance(wcs, center, err);
        set(coordErrKey, skyCov);
    }
}

void SourceRecord::_assign(BaseRecord const &other) {
    try {
        SourceRecord const &s = dynamic_cast<SourceRecord const &>(other);
        _footprint = s._footprint;
    } catch (std::bad_cast &) {
    }
}

std::shared_ptr<SourceTable> SourceTable::make(Schema const &schema,
                                               std::shared_ptr<IdFactory> const &idFactory) {
    if (!checkSchema(schema)) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "Schema for Source must contain at least the keys defined by getMinimalSchema().");
    }
    std::shared_ptr<SourceTable> table(new SourceTable(schema, idFactory));
    table->getSchema().getAliasMap()->setTable(table);
    return table;
}

SourceTable::SourceTable(Schema const &schema, std::shared_ptr<IdFactory> const &idFactory)
        : SimpleTable(schema, idFactory), _slots(schema) {}

SourceTable::SourceTable(SourceTable const &other) = default;

void SourceTable::handleAliasChange(std::string const &alias) {
    if (alias.compare(0, 4, "slot") != 0) {
        return;
    }
    _slots.handleAliasChange(alias, getSchema());
}

SourceTable::MinimalSchema::MinimalSchema() {
    schema = SimpleTable::makeMinimalSchema();
    parent = schema.addField<RecordId>("parent", "unique ID of parent source");
}

SourceTable::MinimalSchema &SourceTable::getMinimalSchema() {
    static MinimalSchema it;
    return it;
}

std::shared_ptr<io::FitsWriter> SourceTable::makeFitsWriter(fits::Fits *fitsfile, int flags) const {
    return std::make_shared<SourceFitsWriter>(fitsfile, flags);
}

std::shared_ptr<BaseTable> SourceTable::_clone() const {
    std::shared_ptr<SourceTable> table(new SourceTable(*this));
    table->getSchema().getAliasMap()->setTable(table);
    return table;
}

std::shared_ptr<BaseRecord> SourceTable::_makeRecord() {
    auto record = constructRecord<SourceRecord>();
    if (getIdFactory()) record->setId((*getIdFactory())());
    return record;
}

template class CatalogT<SourceRecord>;
template class CatalogT<SourceRecord const>;

template class SortedCatalogT<SourceRecord>;
template class SortedCatalogT<SourceRecord const>;
}  // namespace table
}  // namespace afw
}  // namespace lsst
